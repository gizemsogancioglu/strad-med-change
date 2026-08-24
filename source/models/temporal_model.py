import copy
import optuna
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
import matplotlib.pyplot as plt
from source.data_processing.data_model import TemporalEvent, Medication, Features, EventType
from torch.utils.data import DataLoader

from source.data_processing.data_reader import model_path, data_path, results_path
from source.models.base_model import PatientTrajectoryDataset, Evaluator
from transformers import AutoTokenizer
from source.models.model_cfgs.feature_config import (
    ADMISSION_BINARY_COLUMNS, ADMISSION_CONTINUOUS_COLUMNS, ALL_TIME_COLUMNS, FEATURE_GROUP_COLUMNS, LAB_BINARY_COLUMNS, 
    STATIC_CONTINUOUS_COLUMNS, MED_SUMMARY_COLUMNS,
    DIAG_TIME_COLUMNS, CATEGORICAL_COLUMNS, ALL_BOOLEAN_COLUMNS,
    MED_BOOLEAN_FLAG_COLUMNS, NON_NUMERIC_COLUMNS, TIMESTAMPS_CONTINUOUS_COLUMNS,
    active, APPOINTMENT_CONTINUOUS_COLUMNS, TEXT_CONTINUOUS_COLUMNS, TEXT_INTERACTION_CONTINUOUS_COLUMNS, NONE_VALUES
)
from source.models.base_model import PatientTrajectoryDataset, Evaluator, set_seed

from collections import Counter


dutch_bert_model_name = "CLTL/MedRoBERTa.nl"
bert_tokenizer = AutoTokenizer.from_pretrained(dutch_bert_model_name)


# Implementation of TimeLSTM used from : https://github.com/duskybomb/tlstm/blob/master/tlstm.py
# Paper: https://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf
class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        # assumes that batch_first is always true
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, lens, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs

class ModalityAttention(nn.Module):
    def __init__(self, modality_dim, num_modalities):
        super().__init__()
        # one query per modality slot instead of one shared query
        self.query = nn.Parameter(torch.randn(num_modalities, modality_dim))
        self.proj = nn.Linear(modality_dim, modality_dim)
        self.num_modalities = num_modalities

    def forward(self, modality_feats, modality_masks=None):
        stacked = torch.stack(modality_feats, dim=2)   # [B, S, M, D]
        projected = torch.tanh(self.proj(stacked))     # [B, S, M, D]
        # per-modality dot product with its own query row
        attn_scores = torch.einsum('bsmd,md->bsm', projected, self.query)
        if modality_masks is not None:
            mask = torch.stack(modality_masks, dim=2)
            attn_scores = attn_scores.masked_fill(mask.squeeze(-1) == 0, float('-inf'))
        attn_weights = functional.softmax(attn_scores, dim=-1).unsqueeze(-1)
        fused = torch.sum(attn_weights * stacked, dim=2)
        return fused, attn_weights
    
def _make_mlp(in_dim, out_dim, hidden_dim=None, mlp_dropout=0.1):
    hidden_dim = hidden_dim or max(in_dim // 2, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(mlp_dropout),
        nn.Linear(hidden_dim, out_dim),
    )

def _make_proj(in_dim, out_dim, mlp_dropout=0.1):
    """
    Single-hop nonlinear projection: Linear -> ReLU -> Dropout.
    """
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        # nn.ReLU(),
        # nn.Dropout(mlp_dropout),
    )

def _make_mlp_backbone(in_dim, hidden_dim, num_layers=1, dropout=0.1):
    """
    Stack of num_layers Linear -> ReLU -> Dropout blocks, used as the
    'mlp' backbone option.
    """
    layers = []
    cur_dim = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(cur_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        cur_dim = hidden_dim
    return nn.Sequential(*layers)

class DynamicModel(nn.Module):
    def __init__(self, target_task, train_dataset, input_config, model_cfg, modality_dim=128,
             hidden_dim=128, num_layers=2, dropout=0.1, mlp_dropout=0.1,
             diag_embed=32, med_embed=64, static_proj_dim=32, fusion_type=None, use_global_proj=None):
        super().__init__()
        self.input_config = input_config
        self.task = target_task.task
        self.num_labels = target_task.num_labels
        self.model_name = model_cfg['model']
        self.finetune_bert = model_cfg['finetune_bert']
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.modality_dim = modality_dim
        self.dropout_val = dropout
        self.mlp_dropout = mlp_dropout
        self.bert_dim = train_dataset.bert_dim
        self.tfidf_dim = train_dataset.tfidf_dim
        self.modality_encoders = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        self.event_dim_for_concat = train_dataset.event_vocab_size
        self.fusion_type = fusion_type if fusion_type is not None else model_cfg.get('fusion')
        self.use_global_proj   = use_global_proj if use_global_proj is not None else model_cfg.get('use_global_proj', False)
        self.use_modality_proj = modality_dim is not None and self.use_global_proj is False

        self.use_bow_meds      = model_cfg.get('use_bow_meds', True)
        self.use_bow_diag      = model_cfg.get('use_bow_diag', True)

        print(f"  Config: modality_dim={modality_dim}, use_modality_proj={self.use_modality_proj}, "
          f"use_bow_meds={self.use_bow_meds}, fusion={self.fusion_type}")
        print(f"diagnosis_vocab_size: {train_dataset.diagnosis_vocab_size}")
        print(f"diag_proj input dim: {train_dataset.diagnosis_vocab_size * 2 + 2}")

        self.diagnosis_vocab_size = train_dataset.diagnosis_vocab_size
        num_modalities = 0
        self.note_event_id = train_dataset.event2id.get(EventType.NOTE.value, -1)
        self.appointment_event_id = train_dataset.event2id.get(EventType.APPOINTMENT.value, -1)

        self.time_delta_col = None
        if self.model_name == "time-lstm":
            col_names = train_dataset.dynamic_columns_continuous
            self.time_delta_col = col_names.index(TemporalEvent.TIME_SINCE_LAST_EVENT.value)

        # ══════════════════════════════════════════════════════════════════════
        # SINGLE FUSED TEXT MODALITY 
        # ══════════════════════════════════════════════════════════════════════
        self.use_text_symptoms    = Features.SYMPTOM_PREDICTIONS in input_config
        self.use_text_interaction = Features.TEXT_INTERACTION in input_config
        self.use_text_embed       = (Features.BERT in input_config) or (Features.TFIDF in input_config)
        self.text_modality_active = (
            self.use_text_symptoms or self.use_text_interaction or self.use_text_embed
        )

        self.bert = None    
        self.text_proj = None
        self.symptom_vocab_size = train_dataset.symptom_vocab_size
        self.text_dim = 0
        if self.text_modality_active:
            # symptoms component
            if self.use_text_symptoms and train_dataset.symptom_vocab_size > 0:
                self.symptom_dim = train_dataset.symptom_pred_dim * train_dataset.symptom_vocab_size
                #self.symptom_proj = nn.Linear(self.symptom_dim, modality_dim)
                self.text_dim += self.symptom_dim
            # interaction metadata component (role one-hot + continuous)
            if self.use_text_interaction and train_dataset.note_meta_dim > 0:
                self.text_dim += train_dataset.note_meta_dim
            # embedding component (BERT or TF-IDF)
            if self.use_text_embed:
                if Features.BERT in input_config:
                    if self.finetune_bert:
                        self.bert = AutoModel.from_pretrained(dutch_bert_model_name)
                        for param in self.bert.parameters():
                            param.requires_grad = False
                        total_layers = self.bert.config.num_hidden_layers
                        for name, param in self.bert.named_parameters():
                            if f".layer.{total_layers - 1}." in name or "pooler" in name:
                                param.requires_grad = True
                        self.text_input_dim = self.bert.config.hidden_size
                    else:
                        self.text_input_dim = self.bert_dim
                else:  # TFIDF
                    self.text_input_dim = self.tfidf_dim
                self.text_dim += self.text_input_dim
            
            self.text_proj =  _make_proj(self.text_dim, modality_dim) if self.use_modality_proj else None
           
            num_modalities += 1

            print(f"  TEXT modality: symptoms={self.use_text_symptoms}, "
                  f"interaction={self.use_text_interaction} (meta_dim={train_dataset.note_meta_dim}), "
                  f"embed={self.use_text_embed} (in_dim={self.text_dim})")
      
        # ── APPOINTMENT ────────────────────────────────────────────────────────
        if Features.APPOINTMENT in input_config:
            self.appointment_proj =  _make_proj(train_dataset.appointment_dim, modality_dim) \
                if self.use_modality_proj else None
            num_modalities += 1

        if Features.TIMESTAMPS in input_config:
            self.dynamic_proj =  _make_proj(train_dataset.dynamic_dim, modality_dim) \
                if self.use_modality_proj else None
            num_modalities += 1

        if Features.EVENT_TYPE in input_config:
            self.event_proj = _make_proj(train_dataset.event_vocab_size, modality_dim) \
                if self.use_modality_proj else None
            num_modalities += 1

        if Features.TRIGGER_MEDICATIONS in input_config:
            if self.use_modality_proj:
                self.trigger_med_proj = _make_proj(
                    train_dataset.trigger_med_dim + self.event_dim_for_concat, modality_dim)
            else:
                self.trigger_med_proj = None
            num_modalities += 1     
           
        
        if Features.DIAGNOSIS in input_config:
            if not self.use_bow_diag:
                self.diag_embedding_full = nn.Embedding(
                    train_dataset.diagnosis_vocab_size, diag_embed, padding_idx=0)
            else:
                self.diag_embedding_full = None

            if self.use_modality_proj:
                in_dim = (train_dataset.diagnosis_vocab_size * 2 + 2) if self.use_bow_diag \
                         else (diag_embed * 2 + 2)
                in_dim += + self.event_dim_for_concat
                self.diag_proj = _make_proj(in_dim, modality_dim)
            else:
                self.diag_proj = None
            num_modalities += 1
            print(f"  Diag: use_bow_diag={self.use_bow_diag}, "
                f"diag_proj={'set' if getattr(self,'diag_proj',None) is not None else 'None'}, "
                f"diag_embedding={'set' if getattr(self,'diag_embedding_full',None) is not None else 'None'}")


        self.atc_l3_size = (
            len(train_dataset.med_vocab["l3"]) if train_dataset.med_vocab is not None else 0
        )
        self.atc_l4_size = (
            len(train_dataset.med_vocab["l4"]) if train_dataset.med_vocab is not None else 0
        )
        self.trigger_atc_vocab_size = getattr(train_dataset, 'trigger_atc_vocab_size', 0)

        self.has_active_atc_input  = Features.ACTIVE_MEDICATIONS  in input_config and self.atc_l3_size > 0
       
        # active-meds embedding only — trigger ATC is always BOW, no embedding table needed
        self.atc_embedding_l3 = nn.Embedding(self.atc_l3_size, med_embed, padding_idx=0) \
            if (not self.use_bow_meds) and self.has_active_atc_input else None

        self.atc_embedding_l4 = nn.Embedding(self.atc_l4_size, med_embed, padding_idx=0) \
            if (not self.use_bow_meds) and self.has_active_atc_input else None

        self.active_atc_dim  = self.atc_l4_size if self.use_bow_meds else med_embed 
      
        if Features.ACTIVE_MEDICATIONS in input_config:
            med_proj_in_dim = self.active_atc_dim
            if Features.MED_SUMMARY in input_config:
                med_proj_in_dim += train_dataset.med_summary_dim
            med_proj_in_dim += self.event_dim_for_concat
            if self.use_modality_proj:
                self.med_proj = _make_proj(med_proj_in_dim, modality_dim)
            else:
                self.med_proj = None
            num_modalities += 1

                           
        if Features.ADMISSION in input_config:
            self.admission_proj = _make_proj(train_dataset.admission_dim + self.event_dim_for_concat, modality_dim) \
                if self.use_modality_proj else None

            num_modalities += 1

        if Features.LAB_RESULTS in input_config:
            self.lab_proj = _make_proj(train_dataset.lab_dim + self.event_dim_for_concat, modality_dim) \
                if self.use_modality_proj else None
            num_modalities += 1

        if Features.MED_BOOLEAN_FLAGS in input_config:
            self.med_boolean_proj = _make_proj(train_dataset.med_boolean_dim + self.event_dim_for_concat, modality_dim) \
            if self.use_modality_proj else None
            num_modalities += 1


        if Features.PATIENT in input_config:
            self.static_proj = _make_proj(train_dataset.static_dim + self.event_dim_for_concat, modality_dim) \
                if self.use_modality_proj else None
            num_modalities += 1

        
        if Features.PATIENT in input_config:
            print(f"  static_dim   : {train_dataset.static_dim}")
        if Features.LAB_RESULTS in input_config:
            print(f"  lab_dim      : {train_dataset.lab_dim}")
        if Features.ADMISSION in input_config:
            print(f"  admission_dim: {train_dataset.admission_dim}")
       
            
        # ── BACKBONE INPUT SIZE ────────────────────────────────────────────────
        if self.use_modality_proj:
            lstm_input_size = modality_dim if self.fusion_type in ('attention', 'mean_pool') else modality_dim * num_modalities
                
        else:
            _sz = 0
            if Features.ADMISSION          in input_config: _sz += train_dataset.admission_dim + self.event_dim_for_concat
            if Features.TIMESTAMPS         in input_config: _sz += train_dataset.dynamic_dim + self.event_dim_for_concat
            if Features.LAB_RESULTS        in input_config: _sz += train_dataset.lab_dim + self.event_dim_for_concat
            if Features.EVENT_TYPE         in input_config: _sz += train_dataset.event_vocab_size 
            if Features.APPOINTMENT        in input_config: _sz += train_dataset.appointment_dim + self.event_dim_for_concat
            if Features.MED_BOOLEAN_FLAGS  in input_config: _sz += train_dataset.med_boolean_dim + self.event_dim_for_concat
            if Features.TRIGGER_MEDICATIONS in input_config:
                _sz += train_dataset.trigger_med_dim  + self.event_dim_for_concat
            if Features.DIAGNOSIS in input_config:
                _sz += (train_dataset.diagnosis_vocab_size * 2 + 2) + self.event_dim_for_concat if self.use_bow_diag \
                    else (diag_embed * 2 + 2) + self.event_dim_for_concat
            if Features.ACTIVE_MEDICATIONS in input_config:
                _sz += self.active_atc_dim + self.event_dim_for_concat
            if Features.MED_SUMMARY in input_config:
                _sz += train_dataset.med_summary_dim 
            if Features.PATIENT in input_config:
                _sz += train_dataset.static_dim + self.event_dim_for_concat
            if getattr(self, 'text_modality_active', False): 
                _sz += train_dataset.text_dim + self.event_dim_for_concat
            lstm_input_size = _sz
        
        print(f"  → lstm_input_size (sum) = {lstm_input_size}")
            
        if self.use_global_proj:
            self.global_proj = _make_proj(lstm_input_size, hidden_dim)
            lstm_input_size = hidden_dim   # backbone now consumes the projected width
        else:
            self.global_proj = None

        self.pre_backbone_ln = nn.LayerNorm(lstm_input_size) 

        if self.model_name == "time-lstm":
            self.backbone = TimeLSTM(
                input_size=lstm_input_size, hidden_size=hidden_dim,
                cuda_flag=torch.cuda.is_available())
        elif self.model_name.endswith("lstm"):
            self.backbone = nn.LSTM(
                input_size=lstm_input_size, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0)
        elif self.model_name.endswith("gru"):
            self.backbone = nn.GRU(
                input_size=lstm_input_size, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0)
        elif self.model_name == "mlp":
            self.backbone = _make_mlp_backbone(
                lstm_input_size, hidden_dim, num_layers=self.num_layers, dropout=dropout)
       
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.input_proj = nn.Linear(lstm_input_size, hidden_dim)

        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(mlp_dropout)

        out_dim = 1 if self.task == 'regression' else self.num_labels
        self.fc = _make_mlp(hidden_dim, out_dim, mlp_dropout=mlp_dropout)
        
        self.modality_attention = ModalityAttention(
                modality_dim=modality_dim, num_modalities=num_modalities) if self.fusion_type == "attention" else None

    # ------------------------------------------------------------------
    def forward(self, flat_input_ids, flat_attn_masks, chunk_to_batch, chunk_to_event,
            static_feats, dynamic_feats, diag_time, text_vec_feats, event_type_onehot,
            diag_main_ids, diag_secondary_ids, med_summary_feats, med_tensor,
            symptom_feats, note_meta_feats, diag_group_ids, diag_subgroup_ids,
            appointment_feats, admission_feats, lab_feats, med_boolean_feats,
            trigger_med_feats, seq_lengths):
        
        device = dynamic_feats.device
        B, S = dynamic_feats.shape[:2]

        structured_feats_list = []
        modality_masks = []

        event_ids = event_type_onehot.argmax(dim=-1)
        is_note_mask = (event_ids == self.note_event_id).unsqueeze(-1).float()
        is_appt_mask = (event_ids == self.appointment_event_id).unsqueeze(-1).float()
        
        # ── ADMISSION ──────────────────────────────────────────────────────────
        if Features.ADMISSION in self.input_config:
            if self.event_dim_for_concat > 0:  
                admission_feats = torch.cat([admission_feats, event_type_onehot], dim=-1)

            out_adm = self.admission_proj(admission_feats) if self.use_modality_proj else admission_feats
            structured_feats_list.append(out_adm)
            modality_masks.append(torch.ones(B, S, 1, device=device))

        # ── TIMESTAMPS ─────────────────────────────────────────────────────────
        if Features.TIMESTAMPS in self.input_config:
            if self.event_dim_for_concat > 0:  
                dynamic_feats = torch.cat([dynamic_feats, event_type_onehot], dim=-1)
            out_ts = self.dynamic_proj(dynamic_feats) if self.use_modality_proj else dynamic_feats
            structured_feats_list.append(out_ts)
            modality_masks.append(torch.ones(B, S, 1, device=device))

        # ── LAB RESULTS ────────────────────────────────────────────────────────
        if Features.LAB_RESULTS in self.input_config:
            if self.event_dim_for_concat > 0:  
                lab_feats = torch.cat([lab_feats, event_type_onehot], dim=-1)
            out_lab = self.lab_proj(lab_feats) if self.use_modality_proj else lab_feats
            structured_feats_list.append(out_lab)
            modality_masks.append(torch.ones(B, S, 1, device=device))

        # ══════════════════════════════════════════════════════════════════════
        # SINGLE FUSED TEXT MODALITY  (symptoms + interaction + embeddings)
        # all note-event only → masked by is_note_mask, summed in modality_dim
        # ══════════════════════════════════════════════════════════════════════
        if self.text_modality_active:
            text_components = []

            if self.use_text_symptoms:
                oh = functional.one_hot(symptom_feats, num_classes=self.symptom_vocab_size).float()
                oh = oh.flatten(start_dim=2)          # [B,S, num_cols * vocab]
                text_components.append(oh)

            if self.use_text_interaction:
                text_components.append(note_meta_feats)

            if self.use_text_embed:
                if not self.finetune_bert:
                    embed_in = text_vec_feats
                else:
                    embed_in = torch.zeros(B, S, self.text_input_dim, device=device)
                    if flat_input_ids.numel() > 0:
                        flat_input_ids  = flat_input_ids.to(device)
                        flat_attn_masks = flat_attn_masks.to(device)
                        chunk_to_batch  = chunk_to_batch.to(device)
                        chunk_to_event  = chunk_to_event.to(device)
                        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attn_masks)
                        cls_vectors = outputs.last_hidden_state[:, 0, :]
                        chunk_counts = torch.zeros(B, S, 1, device=device)
                        for i in range(cls_vectors.size(0)):
                            b, s = chunk_to_batch[i], chunk_to_event[i]
                            embed_in[b, s] += cls_vectors[i]
                            chunk_counts[b, s] += 1
                        embed_in = embed_in / chunk_counts.clamp(min=1)
                text_components.append(embed_in)

            text_fused = torch.cat(text_components, dim=-1) if text_components \
                         else torch.zeros(B, S, self.text_dim, device=device)
            out_text = self.text_proj(text_fused) if self.use_modality_proj else text_fused

            #text_fused = text_fused * is_note_mask
            structured_feats_list.append(out_text)
            modality_masks.append(torch.ones(B, S, 1, device=device))
            #modality_masks.append(is_note_mask)

        # ── EVENT TYPE ─────────────────────────────────────────────────────────
        if Features.EVENT_TYPE in self.input_config:
            out_event_type = self.event_proj(event_type_onehot) if self.use_modality_proj else event_type_onehot
            structured_feats_list.append(out_event_type)
            modality_masks.append(torch.ones(B, S, 1, device=device))

        # ── APPOINTMENT ────────────────────────────────────────────────────────
        if Features.APPOINTMENT in self.input_config:
            out_appt = self.appointment_proj(appointment_feats) if self.use_modality_proj else appointment_feats
            appt_proj_out = out_appt * is_appt_mask
            structured_feats_list.append(appt_proj_out)
            modality_masks.append(is_appt_mask)

        # ── DIAGNOSIS ──────────────────────────────────────────────────────────
        if Features.DIAGNOSIS in self.input_config:
            t_main = diag_time[:, :, 0:1]
            t_sec  = diag_time[:, :, 1:2]
           
            if self.use_bow_diag:
                main_repr = functional.one_hot(diag_main_ids, num_classes=self.diagnosis_vocab_size).float()
                sec_repr  = functional.one_hot(diag_secondary_ids, num_classes=self.diagnosis_vocab_size).float()
            
            else:
                main_repr = self.diag_embedding_full(diag_main_ids)
                sec_repr  = self.diag_embedding_full(diag_secondary_ids)

            if self.event_dim_for_concat > 0:
                diag_vec = torch.cat([main_repr, t_main, sec_repr, t_sec, event_type_onehot], dim=-1)
            else:
                diag_vec = torch.cat([main_repr, t_main, sec_repr, t_sec], dim=-1)

            if self.use_modality_proj:
                structured_feats_list.append(self.diag_proj(diag_vec))
            else:
                structured_feats_list.append(diag_vec)


            modality_masks.append(torch.ones(B, S, 1, device=device))

        # ── TRIGGER MEDICATIONS ────────────────────────────────────────────────
        if Features.TRIGGER_MEDICATIONS in self.input_config:
            if self.event_dim_for_concat > 0:
                trigger_med_feats = torch.cat([trigger_med_feats, event_type_onehot], dim=-1)         
            out_trigger = self.trigger_med_proj(trigger_med_feats) \
                        if self.use_modality_proj else trigger_med_feats
            structured_feats_list.append(out_trigger)
            modality_masks.append(torch.ones(B, S, 1, device=device))
    
        # ── ACTIVE MEDICATIONS ─────────────────────────────────────────────────
        if Features.ACTIVE_MEDICATIONS in self.input_config:
            med_ids_l3 = med_tensor["l3"].to(device)
            med_ids_l4 = med_tensor["l4"].to(device)
            med_mask_t = med_tensor["mask"].to(device)
            mask_exp   = med_mask_t.unsqueeze(-1)

            if self.use_bow_meds:
                bow_l3 = torch.zeros(B, S, self.atc_l3_size, device=device)
                valid_ids_l3 = med_ids_l3 * med_mask_t.long()
                bow_l3.scatter_add_(2, valid_ids_l3, med_mask_t.float())
                bow_l3[:, :, 0] = 0
                bow_l3[:, :, 1] = 0

                bow_l4 = torch.zeros(B, S, self.atc_l4_size, device=device)
                valid_ids_l4 = med_ids_l4 * med_mask_t.long()
                bow_l4.scatter_add_(2, valid_ids_l4, med_mask_t.float())
                bow_l4[:, :, 0] = 0
                bow_l4[:, :, 1] = 0
                bow_l4 = bow_l4.clamp(max=1.0)

                med_repr = torch.cat([bow_l3, bow_l4], dim=-1)
                med_repr = bow_l4 
            else:
                emb_l3 = self.atc_embedding_l3(med_ids_l3)
                emb_l4 = self.atc_embedding_l4(med_ids_l4)
                pooled_l3 = (emb_l3 * mask_exp).sum(dim=2) / mask_exp.sum(dim=2).clamp(min=1e-6)              
                pooled_l4 = (emb_l4 * mask_exp).sum(dim=2) / mask_exp.sum(dim=2).clamp(min=1e-6)
                pooled = (pooled_l3 + pooled_l4) / 2
                #med_repr = torch.cat([pooled_l3, pooled_l4], dim=-1)  # [B, S, med_embed*2] ✓ matches active_atc_dim
                med_repr = pooled_l4

            if Features.MED_SUMMARY in self.input_config:
                med_repr = torch.cat([med_repr, med_summary_feats], dim=-1)
            if self.event_dim_for_concat > 0:
                med_repr = torch.cat([med_repr, event_type_onehot], dim=-1)
            if self.use_modality_proj:
                structured_feats_list.append(self.med_proj(med_repr))
            else:
                structured_feats_list.append(med_repr)

            modality_masks.append(torch.ones(B, S, 1, device=device))
            del med_ids_l3, med_mask_t
            
        # ── PATIENT ─────────────────────────────────────────────
        if Features.PATIENT in self.input_config:
            static_exp = static_feats.unsqueeze(1).expand(-1, S, -1)
            if self.event_dim_for_concat > 0:
                static_exp = torch.cat([static_exp, event_type_onehot], dim=-1)
            out_static = self.static_proj(static_exp) if self.use_modality_proj else static_exp
            structured_feats_list.append(out_static)
            modality_masks.append(torch.ones(B, S, 1, device=device))
            
        # ── FUSION ─────────────────────────────────────────────────────────────
        attn_weights = None
        if self.fusion_type == "concat":
            fused_feats  = torch.cat(structured_feats_list, dim=-1)
        elif self.fusion_type == "mean_pool":
            stacked = torch.stack(structured_feats_list, dim=2)
            mask    = torch.stack(modality_masks, dim=2)
            fused_feats  = (stacked * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1e-6)
        elif self.fusion_type == 'attention':
            fused_feats, attn_weights = self.modality_attention(
                structured_feats_list, modality_masks)
        else:
            print("Fusion type is not known...")
        
        if self.global_proj is not None:
            fused_feats = self.global_proj(fused_feats)
        fused_feats = self.pre_backbone_ln(fused_feats) 

        # ── TIME DELTAS ────────────────────────────────────────────────────────
        time_deltas = None
        if self.model_name == "time-lstm" and self.time_delta_col is not None:
            time_deltas = dynamic_feats[:, :, self.time_delta_col]
            time_deltas = torch.log1p(time_deltas.clamp(min=0))
        lstm_input = fused_feats

        # ── BACKBONE ───────────────────────────────────────────────────────────
        if self.model_name == 'time-lstm':
            out = self.backbone(lstm_input, time_deltas, seq_lengths)
        elif self.model_name == 'mlp':
            out = self.backbone(lstm_input)
        elif self.model_name.endswith(("lstm", "gru")):
            out, _ = self.backbone(lstm_input)
        else:
            x = self.input_proj(lstm_input)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device).bool()
            pad_mask = torch.arange(S, device=device).expand(B, S) >= seq_lengths.unsqueeze(1)
            out = self.backbone(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)
        
        if seq_lengths is not None:
            pad_mask = (
                torch.arange(S, device=device).expand(B, S) < seq_lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            out = out * pad_mask

        if self.task == 'regression':
            out = out.squeeze() - 1

        if not self.training:
            return out, attn_weights
        return out, None


def check_nan(name, x):
    if torch.isnan(x).any():
        print(f"NaNs detected in {name}")

def collate_fn(batch):
    (
        input_ids_list,
        attention_mask_list,
        static_feats_list,
        dynamic_feats_list,
        diag_time_list,
        text_vec_list,
        event_one_hot_list,
        diag_main_ids_list,
        diag_secondary_ids_list,
        med_summary_list,
        ATC_data_list,
        labels_list,
        note_meta_list,
        diag_group_ids_list,
        diag_subgroup_ids_list,
        symptom_ids_list,
        appointment_feats_list,
        admission_feats_list,      
        lab_feats_list,            
        med_boolean_feats_list,    
        trigger_med_feats_list,          
    ) = zip(*batch)

    # =====================================================
    # PAD TEMPORAL FEATURES
    # =====================================================
    seq_lengths = torch.tensor([len(seq) for seq in dynamic_feats_list], dtype=torch.long)

    pad_fn      = lambda x: pad_sequence([t.float() for t in x], batch_first=True, padding_value=0)
    pad_fn_long = lambda x: pad_sequence([t.long() for t in x],  batch_first=True, padding_value=0)

    static_feats         = torch.stack([x.float() for x in static_feats_list])   # [B, static_dim]
    dynamic_feats        = pad_fn(dynamic_feats_list)                             # [B, S, dynamic_dim]
    med_summary_feats    = pad_fn(med_summary_list)                               # [B, S, med_summary_dim]
    diag_time_feats      = pad_fn(diag_time_list)                                 # [B, S, 2]
    pretrained_text_feats = pad_fn(text_vec_list)                                 # [B, S, text_dim]
    event_one_hot        = pad_fn(event_one_hot_list)                             # [B, S, event_vocab_size]
    diag_main_ids        = pad_fn_long(diag_main_ids_list)                        # [B, S]
    diag_secondary_ids   = pad_fn_long(diag_secondary_ids_list)                   # [B, S]
    symptom_ids          = pad_sequence(
        [s.long() for s in symptom_ids_list], batch_first=True, padding_value=0
    )                                                                              # [B, S, num_symptom_cols]
    note_meta_feats      = pad_fn(note_meta_list)                                 # [B, S, note_meta_dim]
    diag_group_ids       = pad_fn_long(diag_group_ids_list)                       # [B, S]
    diag_subgroup_ids    = pad_fn_long(diag_subgroup_ids_list)                    # [B, S]
    appointment_feats    = pad_fn(appointment_feats_list)                         # [B, S, appointment_dim]

    # NEW modality tensors
    admission_feats      = pad_fn(admission_feats_list)                           # [B, S, admission_dim]
    lab_feats            = pad_fn(lab_feats_list)                                 # [B, S, lab_dim]
    med_boolean_feats    = pad_fn(med_boolean_feats_list)                         # [B, S, med_boolean_dim]

    # handle regression (float) vs classification (long) labels
    is_regression = labels_list[0].dtype == torch.float32
    if is_regression:
        labels = pad_sequence([s.float() for s in labels_list], batch_first=True, padding_value=0.0)
    else:
        labels = pad_sequence([s.long() for s in labels_list], batch_first=True, padding_value=-100)

    # =====================================================
    # PAD MEDICATION TENSORS
    # =====================================================
    B = len(ATC_data_list)
    S = dynamic_feats.size(1)          # align to real padded seq length
    M = max(
        (len(med_ids) for events in ATC_data_list for med_ids in events),
        default=0,
    )

    med_ids_l2 = torch.zeros(B, S, M, dtype=torch.long)
    med_ids_l3 = torch.zeros(B, S, M, dtype=torch.long)
    med_ids_l4 = torch.zeros(B, S, M, dtype=torch.long)
    med_ids_l5 = torch.zeros(B, S, M, dtype=torch.long)
    med_mask   = torch.zeros(B, S, M, dtype=torch.float)

    for b, events in enumerate(ATC_data_list):
        for s, med_ids in enumerate(events):
            for m, med in enumerate(med_ids):
                med_ids_l2[b, s, m] = med["l2"]
                med_ids_l3[b, s, m] = med["l3"]
                med_ids_l4[b, s, m] = med["l4"]
                med_ids_l5[b, s, m] = med["l5"]
                med_mask[b, s, m]   = 1.0

    med_tensor = {
        "l2":       med_ids_l2,   # [B, S, M]
        "l3":       med_ids_l3,
        "l4":       med_ids_l4,
        "l5":       med_ids_l5,
        #"duration": durations,    # [B, S, M]
        "mask":     med_mask,     # [B, S, M]
    }

    trigger_med_feats    = pad_fn(trigger_med_feats_list)
   
    # =====================================================
    # EVENT MASK (for loss computation)
    # =====================================================
    if is_regression:
        mask = (labels != 0.0).float()
    else:
        mask = (labels != -100).float()

    # =====================================================
    # BERT CHUNKS (flattened for efficient batched encoding)
    # =====================================================
    flat_input_ids      = []
    flat_attention_masks = []
    chunk_to_batch      = []
    chunk_to_event      = []

    for b, sample_input_ids in enumerate(input_ids_list):
        for s, (event_chunks_ids, event_chunks_masks) in enumerate(
                zip(sample_input_ids, attention_mask_list[b])
        ):
            for chunk_ids, chunk_mask in zip(event_chunks_ids, event_chunks_masks):
                flat_input_ids.append(chunk_ids)
                flat_attention_masks.append(chunk_mask)
                chunk_to_batch.append(b)
                chunk_to_event.append(s)

    if len(flat_input_ids) > 0:
        flat_input_ids   = torch.stack(flat_input_ids)
        flat_attn_masks  = torch.stack(flat_attention_masks)
        chunk_to_batch   = torch.tensor(chunk_to_batch, dtype=torch.long)
        chunk_to_event   = torch.tensor(chunk_to_event, dtype=torch.long)
    else:
        flat_input_ids   = torch.empty(0, dtype=torch.long)
        flat_attn_masks  = torch.empty(0, dtype=torch.long)
        chunk_to_batch   = torch.empty(0, dtype=torch.long)
        chunk_to_event   = torch.empty(0, dtype=torch.long)

    return [
        flat_input_ids,          # 0
        flat_attn_masks,         # 1
        chunk_to_batch,          # 2
        chunk_to_event,          # 3
        static_feats,            # 4
        dynamic_feats,           # 5
        diag_time_feats,         # 6
        pretrained_text_feats,   # 7
        event_one_hot,           # 8
        diag_main_ids,           # 9
        diag_secondary_ids,      # 10
        med_summary_feats,       # 11
        med_tensor,              # 12
        symptom_ids,             # 13
        note_meta_feats,         # 14
        diag_group_ids,          # 15
        diag_subgroup_ids,       # 16
        appointment_feats,       # 17
        admission_feats,         # 18  
        lab_feats,               # 19  
        med_boolean_feats,       # 20  
        trigger_med_feats,      # 21
        labels,                 # 22
        seq_lengths,            # 23
        mask,                   # 24
    ]
def build_optimizer(model, optim_cfg):
    bert_params = [p for n, p in model.named_parameters()
                   if "bert" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if "bert" not in n and p.requires_grad]

    wd = optim_cfg.get("weight_decay", 1e-4)

    if optim_cfg["optimizer"] == "adamw":
        return optim.AdamW([
            {"params": bert_params, "lr": optim_cfg["lr"] * 0.1},  # 10x smaller for BERT
            {"params": other_params, "lr": optim_cfg["lr"]},
        ], weight_decay=wd)
    else:
        return optim.Adam([
            {"params": bert_params, "lr": optim_cfg["lr"] * 0.1},
            {"params": other_params, "lr": optim_cfg["lr"]},
        ], weight_decay=wd)


def build_vocab(df, columns):
    """
    Build a mapping from unique values in the given column(s) to integer IDs.

    All empty/missing/unknown values are excluded before building the vocab
    "

    Args:
        df      : pd.DataFrame
        columns : str or list of column names

    Returns:
        vocab : dict mapping value -> id, real tokens only, IDs from 0
    """
    if isinstance(columns, str):
        columns = [columns]

    unique_values = set()
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].apply(lambda v: "nan" if pd.isna(v) else str(v))
        unique_values.update(vals.unique())
       

    unique_values = sorted(v for v in unique_values if v not in NONE_VALUES)

    return {val: idx for idx, val in enumerate(unique_values)}

def build_vocab_min_df(train_df, atc_col, nmbr_atc_code, min_df=50):
    """
    Document-frequency-pruned vocab, fit on TRAIN ONLY.
    """
    df_counts = Counter()
    for atc in train_df[atc_col].fillna(""):
        if not isinstance(atc, str) or not atc.strip():
            continue
        l_in_row = {
            c.strip()[:nmbr_atc_code]
            for c in atc.split()
            if c.strip() and c.strip() not in NONE_VALUES
        }
        df_counts.update(l_in_row)

    kept = sorted(code for code, n in df_counts.items() if n >= min_df)
    vocab = {"<PAD>": 0, "<NONE>": 1}
    vocab.update({code: i + 2 for i, code in enumerate(kept)})
    print(f"  L_{nmbr_atc_code} min_df={min_df}: kept {len(kept)} / {len(df_counts)} codes "
          f"(dropped {len(df_counts) - len(kept)} below threshold)")
    return vocab


def build_diag_hierarchy_vocabs(df, diag_columns):
    """
    Build group and subgroup vocab dicts from DSM diagnosis codes.
    Accepts a list of columns (e.g. main + secondary diagnosis).

    All empty/missing/unknown values are normalised to "<NONE>" before
    building the vocab, then excluded — they map to <NONE>=1 at lookup time.

    Token convention: <PAD>=0, <NONE>=1, real tokens from 2.

    e.g. "D5_4.02.01" -> group "D5_4", subgroup "D5_4.02"

    Returns
    -------
    diag_group2id    : {"<PAD>": 0, "<NONE>": 1, "D5_4": 2, ...}
    diag_subgroup2id : {"<PAD>": 0, "<NONE>": 1, "D5_4.02": 2, ...}
    """
    if isinstance(diag_columns, str):
        diag_columns = [diag_columns]

    # all values meaning "absent" — normalised before exclusion
  
    groups    = set()
    subgroups = set()

    for col in diag_columns:
        vals = df[col].apply(lambda v: "nan" if pd.isna(v) else str(v))
        for code in vals.unique():
            if code in NONE_VALUES:
                continue
            parts = str(code).split(".")
            groups.add(parts[0])
            if len(parts) >= 2:
                subgroups.add(f"{parts[0]}.{parts[1]}")

    diag_group2id = {"<PAD>": 0, "<NONE>": 1}
    diag_group2id.update({g: i + 2 for i, g in enumerate(sorted(groups))})

    diag_subgroup2id = {"<PAD>": 0, "<NONE>": 1}
    diag_subgroup2id.update({s: i + 2 for i, s in enumerate(sorted(subgroups))})

    return diag_group2id, diag_subgroup2id

def get_trajectory_datasets(train_df, val_df, test_df, target_task, model_cfg):
    """
    Prepares patient trajectory datasets for training, validation, and testing.

    Token convention (consistent across all vocabs)
    ----------------
    <PAD>   = 0  : sequence padding
    <NONE>  = 1  : value absent/unknown — diagnosis "UNKNOWN", empty role, no medication
    real tokens start at 2

    All vocabs built from train only. Vocab is None if column absent in train_df.
    build_vocab normalises all absent/unknown values to <NONE> before building,
    so no manual pre-filtering is needed before calling it.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
    target_task : TargetTask
    model_cfg : dict

    Returns
    -------
    train_dataset, val_dataset, test_dataset : PatientTrajectoryDataset
    """
    def col_exists(df, col):
        return col in df.columns

    def add_special_tokens(vocab):
        """<PAD>=0, <NONE>=1, real tokens from 2."""
        vocab = {k: v + 2 for k, v in vocab.items()}
        vocab["<PAD>"]  = 0
        vocab["<NONE>"] = 1
        return vocab

    # ---------------------------------------------------------------
    # Events — no missingness, <PAD>=0, real tokens from 1
    # ---------------------------------------------------------------
    event2id = build_vocab(train_df, TemporalEvent.TYPE.value)
    event2id = {k: v + 1 for k, v in event2id.items()}
    event2id["<PAD>"] = 0


    # ---------------------------------------------------------------
    # Gender — no missingness, <PAD>=0, real tokens from 1
    # ---------------------------------------------------------------
    gender2id = build_vocab(train_df, TemporalEvent.PATIENT_GENDER.value)
    gender2id = {k: v + 1 for k, v in gender2id.items()}
    gender2id["<NONE>"] = 0

    # ---------------------------------------------------------------
    # Diagnosis full codes
    # "UNKNOWN" normalised to <NONE> inside build_vocab — no pre-filtering needed
    # ---------------------------------------------------------------
    diag_main_col = TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value
    diag_sec_col  = TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value

    diag_cols_available = [
        c for c in [diag_main_col, diag_sec_col]
        if col_exists(train_df, c)
    ]

    if diag_cols_available:
        diagnosis2id = add_special_tokens(
            build_vocab(train_df, diag_cols_available)
        )
        diag_group2id, diag_subgroup2id = build_diag_hierarchy_vocabs(
            train_df, diag_columns=diag_cols_available
        )
    else:
        diagnosis2id     = None
        diag_group2id    = None
        diag_subgroup2id = None

    # ---------------------------------------------------------------
    # ── Trigger medications ────────────────────────────────────────────────────
    trigger_atc_col     = TemporalEvent.TRIGGER_MED_ATC_CODE.value #
    trigger_type_col    = TemporalEvent.TRIGGER_MED_DRUG_CLASS.value # ssri, tca, etc. 
    prev_atc_col        = TemporalEvent.PREV_MED_ATC_CODE.value
    prev_drug_class_col = TemporalEvent.PREV_DRUG_CLASS.value

    trigger_atc_vocab = add_special_tokens(
        build_vocab(train_df, trigger_atc_col)
    ) if col_exists(train_df, trigger_atc_col) else None

    prev_atc_vocab = add_special_tokens(
        build_vocab(train_df, prev_atc_col)
    ) if col_exists(train_df, prev_atc_col) else None

    trigger_med_type2id = add_special_tokens(
        build_vocab(train_df, trigger_type_col)
    ) if col_exists(train_df, trigger_type_col) else None

    prev_drug_class2id = add_special_tokens(
        build_vocab(train_df, prev_drug_class_col)
    ) if col_exists(train_df, prev_drug_class_col) else None

    trigger_action_type2id = add_special_tokens(
    build_vocab(train_df, TemporalEvent.TRIGGER_MED_ACTION_TYPE.value)
    ) if col_exists(train_df, TemporalEvent.TRIGGER_MED_ACTION_TYPE.value) else None


    # ---------------------------------------------------------------
    # Note author role
    # empty strings normalised to <NONE> inside build_vocab — no pre-filtering needed
    # ---------------------------------------------------------------
    role2id = add_special_tokens(
        build_vocab(train_df, TemporalEvent.NOTE_CREATION_EMPLOYEE_ROLE.value)
    ) if col_exists(train_df, TemporalEvent.NOTE_CREATION_EMPLOYEE_ROLE.value) else None

    note_type2id = add_special_tokens(
    build_vocab(train_df, TemporalEvent.NOTE_TYPE.value)
    ) if col_exists(train_df, TemporalEvent.NOTE_TYPE.value) else None


    # ---------------------------------------------------------------
    # Symptom progression status - can be missing, nan for non-note events, and excluded note types.
    # empty strings normalised to <NONE> inside build_vocab — no pre-filtering needed
    # ---------------------------------------------------------------
    symptom_columns = [col for col in train_df.columns if col.startswith('symptom_pred_cat_')]

    symptom2id = add_special_tokens(
        build_vocab(train_df, symptom_columns)
    ) if symptom_columns else None

    # appointment_role2id
    appointment_role2id = add_special_tokens(
    build_vocab(train_df, TemporalEvent.APPOINTMENT_ROLE.value)
    ) if col_exists(train_df, TemporalEvent.APPOINTMENT_ROLE.value) else None

    # ── Lab categorical features ───────────────────────────────────────────────
    lab_measure_col   = TemporalEvent.RECENT_LAB_MEASURE.value
    lab_test_type_col = TemporalEvent.RECENT_LAB_TEST_TYPE.value

    lab_measure2id = add_special_tokens(
        build_vocab(train_df, lab_measure_col)
    ) if col_exists(train_df, lab_measure_col) else None

    lab_test_type2id = add_special_tokens(
        build_vocab(train_df, lab_test_type_col)
    ) if col_exists(train_df, lab_test_type_col) else None
    # ---------------------------------------------------------------
    # Medications — ATC hierarchy levels built from train only
    # missing ATC maps to <NONE>=1
    # ---------------------------------------------------------------
    atc_col = Medication.ATC_CODE.value

    if col_exists(train_df, atc_col):
        def all_codes(atc):
            if not isinstance(atc, str) or atc.strip() == "":
                return []
            return [c.strip() for c in atc.split() if c.strip() and c.strip() not in NONE_VALUES]

        def levels_from_df(df, level_slice):
            # collect every code's level-substring across all events
            vals = []
            for atc in df[atc_col]:
                for c in all_codes(atc):
                    vals.append(c[:level_slice] if level_slice else c)
            return pd.DataFrame({"_tmp": vals or ["<NONE>"]})

        med_vocab = {
            "l2": add_special_tokens(build_vocab(levels_from_df(train_df, 3), "_tmp")),
            "l3": build_vocab_min_df(train_df, atc_col, 4, min_df=50),   # ← pruned
            "l4": build_vocab_min_df(train_df, atc_col, 5, min_df=50),   # ← pruned
            "l5": add_special_tokens(build_vocab(levels_from_df(train_df, None), "_tmp")),
        }

    else:
        med_vocab = None
   
    # ---------------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------------
    tokenizer = bert_tokenizer if model_cfg['finetune_bert'] else None

    # ---------------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------------
    shared_kwargs = dict(
        event2id         = event2id,
        gender2id        = gender2id,
        diagnosis2id     = diagnosis2id,
        symptom2id       = symptom2id,  
        med_vocab        = med_vocab,
        bert_tokenizer   = tokenizer,
        diag_group2id    = diag_group2id,
        diag_subgroup2id = diag_subgroup2id,
        role2id          = role2id,
        note_type2id     = note_type2id,
        appointment_role2id = appointment_role2id,
        
    )

    shared_kwargs.update({
    'trigger_med_type2id':      trigger_med_type2id,
    'prev_drug_class2id':       prev_drug_class2id,
    'lab_measure2id':   lab_measure2id,
    'lab_test_type2id': lab_test_type2id,
    })
   

    train_dataset = PatientTrajectoryDataset(
        train_df, target_task,
        **shared_kwargs,
        fit_scaler=True,
    )
    val_dataset = PatientTrajectoryDataset(
        val_df, target_task,
        **shared_kwargs,
        scaler=train_dataset.preprocessors,
    )
    test_dataset = PatientTrajectoryDataset(
        test_df, target_task,
        **shared_kwargs,
        scaler=train_dataset.preprocessors,
    ) if test_df is not None else None

    labels_all = train_df[target_task.outcome].to_numpy()
    # filter to valid med-event labels (not NaN / not -100)
    valid = ~pd.isna(labels_all)
    labels_valid = labels_all[valid]
    labels_valid = labels_valid[labels_valid != -100]

    n_pos = (labels_valid == 1).sum()
    n_neg = (labels_valid == 0).sum()
    total = n_pos + n_neg
    train_dataset.class_weights = (
        [total / (2.0 * n_neg) if n_neg > 0 else 1.0,
        total / (2.0 * n_pos) if n_pos > 0 else 1.0]
    )

    return train_dataset, val_dataset, test_dataset

def summarize_across_seeds(all_results, group_keys=('learner', 'split', 'event_type')):
    """
    Collapse per-seed result rows into mean / std / n_seeds per
    (learner, split, event_type). Returns one row per group with
    <metric>_mean, <metric>_std, n_seeds, and a <metric>_fmt 'mean ± std' string
    for every numeric metric — metric-agnostic, so it works whatever
    task.str_measure() is named.
    """
    keys = [k for k in group_keys if k in all_results.columns]
    numeric_cols = [c for c in all_results.select_dtypes(include='number').columns
                    if c != 'seed']

    grp  = all_results.groupby(keys, dropna=False)
    mean = grp[numeric_cols].mean().add_suffix('_mean')
    std  = grp[numeric_cols].std(ddof=1).add_suffix('_std')
    n    = grp.size().rename('n_seeds')

    summary = mean.join(std).join(n).reset_index()

    for m in numeric_cols:
        mcol, scol = f'{m}_mean', f'{m}_std'
        summary[f'{m}_fmt'] = summary.apply(
            lambda r: f"{r[mcol]:.3f} ± {(0.0 if pd.isna(r[scol]) else r[scol]):.3f}",
            axis=1,
        )
    return summary

def train_backbone_fixed_arch(target_task, config, model_cfg, train_df, val_df, test_df,
                                seeds, target_event_types, epochs=60, device='cuda',
                                save_dir=None):
    """
    Runs Bayesian search ONCE for this backbone, then retrains the winning
    config across multiple seeds and evaluates each on the test set.

    Returns
    -------
    all_results : pd.DataFrame
        One row per seed with aggregated test metrics (BA, AUC, loss, etc.),
        plus a 'seed' column and the winning architecture/optim hyperparams
        attached as columns for traceability.
    all_preds : pd.DataFrame
        Event-level predictions for every seed, with a 'seed' column —
        this is what you feed into the paired cluster bootstrap later.
    best_architecture, best_optim, best_batch_size : the winning config
        found by the single Bayesian search (shared across all seeds).
    """
    train_dataset, val_dataset, test_dataset = get_trajectory_datasets(
        train_df, val_df, test_df, target_task, model_cfg)
    valid_event_ids = [train_dataset.event2id[e] for e in target_event_types
                        if e in train_dataset.event2id] if target_event_types else None

    # ── search ONCE ───────────────────────────────────────────────────────
    _, best_architecture, best_optim, best_batch_size, search_val_score = bayesian_search(
        target_task, config, model_cfg, train_dataset, val_dataset,
        valid_event_ids=valid_event_ids, n_trials=30,
        search_epochs=40, final_epochs=None, device=device)

    print(f"[{model_cfg['model']}] Search done. Best val score={search_val_score:.4f}")
    print(f"[{model_cfg['model']}] Architecture: {best_architecture}")
    print(f"[{model_cfg['model']}] Optim: {best_optim}, batch_size={best_batch_size}")

    val_loader = DataLoader(val_dataset, batch_size=best_batch_size,
                             shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size,
                              shuffle=False, collate_fn=collate_fn)

    per_seed_preds = []
    per_seed_results = []

    for seed in seeds:
        print(f"\n{'='*60}\n[{model_cfg['model']}] SEED {seed}\n{'='*60}")
        set_seed(seed)

        val_score, model_state = train_one_config(
            target_task, config, model_cfg, train_dataset.event2id,
            best_architecture, best_optim, train_dataset, val_dataset,
            valid_event_ids, epochs=epochs, batch_size=best_batch_size, device=device)

        model = DynamicModel(target_task, train_dataset, config, model_cfg,
                              **best_architecture).to(device)
        model.load_state_dict(model_state)

        # ── validation-set eval (for record keeping / sanity check) ────────
        df_val_results, df_val_preds = eval_model(
            model, train_dataset.event2id, target_task, config,
            val_loader, val_df, f"val_seed{seed}",
            valid_event_ids, target_event_types, device)

        # ── test-set eval (what you'll use for the significance test) ──────
        df_test_results, df_test_preds = eval_model(
            model, train_dataset.event2id, target_task, config,
            test_loader, test_df, f"test_seed{seed}",
            valid_event_ids, target_event_types, device)

        df_val_preds["seed"] = seed
        df_test_preds["seed"] = seed
        df_val_results["seed"] = seed
        df_test_results["seed"] = seed

        per_seed_preds.append(df_val_preds)
        per_seed_preds.append(df_test_preds)
        per_seed_results.append(df_val_results)
        per_seed_results.append(df_test_results)

        # persist the raw model_state too, in case you need to reload later
        if save_dir is not None:
            torch.save(
                {"model_state": model_state, "architecture": best_architecture,
                 "optim_cfg": best_optim, "val_score": val_score, "seed": seed},
                f"{save_dir}/{model_cfg['model']}_seed{seed}.pt"
            )

    all_preds = pd.concat(per_seed_preds, ignore_index=True)
    all_results = pd.concat(per_seed_results, ignore_index=True)

    # attach the shared winning hyperparams to every row for traceability
    for k, v in best_architecture.items():
        all_results[f"arch_{k}"] = v
    for k, v in best_optim.items():
        all_results[f"optim_{k}"] = v
    all_results["backbone"] = model_cfg["model"]
    all_preds["backbone"] = model_cfg["model"]

    # summarize mean ± std across seeds, split by split/event_type
    summary = summarize_across_seeds(all_results, group_keys=('split', 'event_type'))

    if save_dir is not None:
        all_results.to_csv(f"{save_dir}/{model_cfg['model']}_all_results.csv", index=False)
        all_preds.to_csv(f"{save_dir}/{model_cfg['model']}_all_preds.csv", index=False)
        summary.to_csv(f"{save_dir}/{model_cfg['model']}_summary.csv", index=False)

    return all_results, all_preds, summary, best_architecture, best_optim, best_batch_size


def train_temporal_model_multiseed(
        target_task, config, model_cfg, train_df, val_df, test_df=None,
        seeds=(13, 42, 123), epochs=60, batch_size=16,
        target_event_types=None, device="cuda"):
    """
    Run train_temporal_model once per seed on a FIXED split.
    Only model init / dropout / train-shuffle vary across seeds.

    Returns
    -------
    all_results : per-seed result rows (every seed, every event_type/split) + 'seed' col
    all_preds   : per-seed event-level predictions + 'seed' col
    summary     : one row per (learner, split, event_type) with mean ± std across seeds
    """
    per_seed_results, per_seed_preds = [], []

    for seed in seeds:
        print(f"\n{'#'*72}")
        print(f"# SEED {seed} | model={model_cfg['model']} | fusion={model_cfg['fusion']}")
        print(f"{'#'*72}")
        set_seed(seed)

        results_df, preds_df = train_temporal_model(
            target_task=target_task, config=config, model_cfg=model_cfg,
            train_df=train_df, val_df=val_df, test_df=test_df,
            epochs=epochs, batch_size=batch_size,
            target_event_types=target_event_types, device=device,
        )

        results_df = results_df.copy(); results_df['seed'] = seed
        preds_df   = preds_df.copy();   preds_df['seed']   = seed
        per_seed_results.append(results_df)
        per_seed_preds.append(preds_df)

    all_results = pd.concat(per_seed_results, ignore_index=True)
    all_preds   = pd.concat(per_seed_preds,   ignore_index=True)
    summary     = summarize_across_seeds(all_results)
    return all_results, all_preds, summary


def train_temporal_model(target_task, config, model_cfg, train_df, val_df, test_df=None, epochs=60,
                         batch_size=16, target_event_types=None, device="cuda"):

    model_name = model_cfg['model']
    train_dataset, val_dataset, test_dataset = get_trajectory_datasets(train_df, val_df, test_df, target_task,
                                                                       model_cfg)
    
    val_loader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
     
    valid_event_ids = None
    if target_event_types is not None:
        valid_event_ids = [train_dataset.event2id[etype] for etype in target_event_types if
                           etype in train_dataset.event2id]

    best_architecture = {
        "modality_dim": 32, #32,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "mlp_dropout": 0.1,
        "diag_embed": 32,
        "med_embed": 32,
        "static_proj_dim": 16,
        "fusion_type": "concat",
    }

    best_optim = {
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-3,
    }
    best_batch_size = batch_size   

    print("Training fixed ablation config (no search)...")
    best_value, best_model_state = train_one_config(
        target_task=target_task, config=config, model_cfg=model_cfg,
        event2id=train_dataset.event2id,
        architecture_params=best_architecture, optim_cfg=best_optim,
        train_dataset=train_dataset, val_dataset=val_dataset,
        valid_event_ids=valid_event_ids,
        epochs=epochs, batch_size=best_batch_size,
        device=device,
    )
    print(f"Ablation training complete: BA={best_value:.4f}")

    ################## TEST ##########################################################################################

    model = DynamicModel(
        target_task,
        train_dataset,
        input_config=config,
        model_cfg=model_cfg,
        **best_architecture
    ).to(device)
    model.load_state_dict(best_model_state)

    df_val, df_val_preds = eval_model(
        model, train_dataset.event2id, target_task, config,
        val_loader, val_df,
        "val_best_overall",
        valid_event_ids=valid_event_ids,
        target_event_types=target_event_types,
        device=device
    )

    final_results = [df_val]
    final_predictions = [df_val_preds]

    if test_df is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, collate_fn=collate_fn)

        df_test, df_test_preds = eval_model(
            model, train_dataset.event2id, target_task, config,
            test_loader, test_df,
            "test_best_overall",
            valid_event_ids=valid_event_ids,
            target_event_types=target_event_types,
            device=device
        )

        final_results.append(df_test)
        final_predictions.append(df_test_preds)

    model_name = model_cfg['model']
    fusion = model_cfg['fusion']
    torch.save(
        {
            "model_state": best_model_state,
            "architecture": best_architecture,
            "optim_cfg": best_optim,
            "val_score": best_value,
        },
        f"{model_path}/best_{model_name}_{fusion}_{target_task.outcome}.pt"
    )
    
    results_df = pd.concat(final_results, ignore_index=True)
    results_df['modality_dim']  = best_architecture['modality_dim']
    results_df['hidden_dim']    = best_architecture['hidden_dim']
    results_df['num_layers']    = best_architecture['num_layers']
    results_df['dropout']       = best_architecture['dropout']
    results_df['lr']            = best_optim['lr']
    results_df['weight_decay']  = best_optim['weight_decay']
    results_df['batch_size']    = best_batch_size
    return (
        results_df,
        pd.concat(final_predictions, ignore_index=True)
    )


def bayesian_search(target_task, config, model_cfg, train_dataset, val_dataset,
                    valid_event_ids=None, n_trials=40, search_epochs=40,
                    final_epochs=None, device="cuda", search_seed=0):

    best_model_state = None
    best_architecture = None
    best_optim = None
    best_batch_size = 16
    best_score = -float("inf")

    def objective(trial):
        nonlocal best_model_state, best_architecture, best_optim, best_batch_size, best_score
        set_seed(search_seed)
        architecture_params = {
            "hidden_dim":      trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers":      trial.suggest_int("num_layers", 1, 3),
            "dropout":         trial.suggest_float("dropout", 0.1, 0.5),
            "mlp_dropout":     trial.suggest_float("mlp_dropout", 0.1, 0.5),
            "med_embed":       trial.suggest_categorical("med_embed", [16, 32, 64, 128]),
            "use_global_proj": trial.suggest_categorical("use_global_proj", [False]),
        }

        if not architecture_params["use_global_proj"]:
            # modality projection is done only if not global projection; otherwise, the global projection is used instead
            architecture_params["modality_dim"] = trial.suggest_categorical("modality_dim", [16, 32, 64, 128])
            architecture_params["fusion_type"] = trial.suggest_categorical("fusion_type", ["concat"])
        else:
            architecture_params["modality_dim"] = None
            architecture_params["fusion_type"] = "concat"  # fusion type is irrelevant if global projection is used

        optim_cfg = {
            "optimizer":    "adamw",
            "lr":           trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        }

        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

        score, model_state = train_one_config(
            target_task=target_task, config=config, model_cfg=model_cfg,
            event2id=train_dataset.event2id,
            architecture_params=architecture_params, optim_cfg=optim_cfg,
            train_dataset=train_dataset, val_dataset=val_dataset,
            valid_event_ids=valid_event_ids,
            device=device, epochs=search_epochs, batch_size=batch_size,
        )

        if score > best_score:
            best_score        = score
            best_model_state  = model_state
            best_architecture = architecture_params
            best_optim        = optim_cfg
            best_batch_size   = batch_size

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial (search, {search_epochs} epochs): BA={best_score:.4f}")
    print(f"Best architecture: {best_architecture}")
    print(f"Best optim: {best_optim}")
    print(f"Best batch_size: {best_batch_size}")

    # ── Optional full-budget retrain, using the EXACT winning config ───────
    # This is what guarantees the returned model matches best_architecture/
    # best_optim/best_batch_size — no separate caller-side retrain needed.
    if final_epochs is not None and final_epochs != search_epochs:
        print(f"\nRetraining winning config at full budget ({final_epochs} epochs, "
              f"batch_size={best_batch_size})...")
        final_score, final_model_state = train_one_config(
            target_task=target_task, config=config, model_cfg=model_cfg,
            event2id=train_dataset.event2id,
            architecture_params=best_architecture, optim_cfg=best_optim,
            train_dataset=train_dataset, val_dataset=val_dataset,
            valid_event_ids=valid_event_ids,
            device=device, epochs=final_epochs, batch_size=best_batch_size,
        )
        print(f"Final retrain score: BA={final_score:.4f} "
              f"(search score was {best_score:.4f})")
        best_model_state = final_model_state
        best_score = final_score

    return best_model_state, best_architecture, best_optim, best_batch_size, best_score


def compute_event_mask(event_one_hot, outcome, valid_event_ids=None, device='cuda'):
    # Flatten everything to 1D for consistent indexing
    event_ids_flat = event_one_hot.argmax(dim=-1).view(-1)
    outcome_flat = outcome.view(-1)

    # Always filter out padding/ignored labels
    mask = (outcome_flat != -100)

    # ONLY filter by event type if valid_event_ids was actually provided
    if valid_event_ids is not None:
        if not isinstance(valid_event_ids, torch.Tensor):
            valid_event_ids = torch.tensor(valid_event_ids, device=device)

        event_filter = torch.isin(event_ids_flat, valid_event_ids)
        mask = mask & event_filter  # Intersection of "not padding" AND "correct type"

    return mask

def train_one_config(target_task, config, model_cfg, event2id, architecture_params, optim_cfg, train_dataset, val_dataset, valid_event_ids,
                     epochs=200,
                     batch_size=1, device="cuda"):
    """
    Train and evaluate a LSTM (or similar) model on patient trajectories.

    Supports either train/val/test split (if test_df is provided) or cross-validation (if test_df=None).

    Parameters
    ----------
    target_task : class Task
        Target task object containing task type ('regression' or 'classification') and labels info.
    config : list
        Input feature configuration.
    model_cfg : dict
        Model hyperparameters.
    train_dataset, val_dataset : Dataset or None
        Dataframes containing patient trajectories.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Batch size for DataLoader.
    device : str
        Device to run training on ('cuda' or 'cpu').

    Returns
    -------
    final_results_df : pd.DataFrame
        Concatenated evaluation results across splits.
    final_predictions_df : pd.DataFrame
        Concatenated model predictions across splits.
    """

    train_loader = DataLoader(
    train_dataset, batch_size=batch_size, 
    shuffle=True,   # ← critical
    collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False,  # ← keep val deterministic
        collate_fn=collate_fn
    )

    model_instance = DynamicModel(
        target_task,
        train_dataset,
        input_config=config,
        model_cfg=model_cfg,
        **architecture_params
    ).to(device)

    print("\n" + "="*60)
    print(f"TRAINING: {model_cfg['model'].upper()} | fusion={model_cfg['fusion']}")
    print(f"Active features: {[f.value for f in config]}")
    print(f"Num modalities : {model_instance.modality_attention.num_modalities if model_instance.modality_attention else 'N/A (concat)'}")
    print(f"LSTM input size: {model_instance.backbone.input_size if hasattr(model_instance.backbone, 'input_size') else 'see model'}")
    print("="*60 + "\n")


    # ── ACTIVE MEDICATIONS ──────────────────────────────────────────────────
    print(f"[LSTM] has_active_atc_input : {model_instance.has_active_atc_input}")
    print(f"[LSTM] atc_l3_size (med_vocab['l3'], pruned min_df=50): {model_instance.atc_l3_size}")
    print(f"[LSTM] active_atc_dim (modality width, incl. count_feat if embed mode): {model_instance.active_atc_dim}")
    if train_dataset.med_vocab is not None:
        print(f"[LSTM]   med_vocab['l3'] keys (first 10): {list(train_dataset.med_vocab['l3'].keys())[:10]}")
    else:
        print(f"[LSTM]   med_vocab is None — active meds column not present / ACTIVE_MEDICATIONS not enabled")

    # ── TRIGGER MEDICATIONS ─────────────────────────────────────────────────
    print(f"[LSTM] has_prev_drug_class: {train_dataset.has_prev_drug_class}")
    print(f"[LSTM] trigger_med_dim (total): {train_dataset.trigger_med_dim}")
    print(f"[LSTM]   drug_class2id : {len(train_dataset.trigger_med_type2id) if train_dataset.trigger_med_type2id else 0}")
    print(f"[LSTM]   prev_drug_class2id (if included): {len(train_dataset.prev_drug_class2id) if train_dataset.prev_drug_class2id else 0}")

    optimizer = build_optimizer(model_instance, optim_cfg)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max",   # maximize score, not minimize loss
    patience=10, 
    factor=0.5
)
    # -------------------------------------------------
    # Loss function
    # -------------------------------------------------
   
    criterion = nn.L1Loss(reduction='none') if target_task.task == 'regression' else nn.CrossEntropyLoss(
            ignore_index=-100, reduction='none')
    
    # --- Tracking variables ---
    best_val_score = -float("inf")
    best_model_state = None
    patience_counter = 0
    patience = 20

    # --- Epoch loop ---
    for epoch in range(epochs):
        model_instance.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        total_train_loss = 0
        n_trained_batches = 0  # ← track only batches that contributed

        for batch in train_loader:
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            (*inputs, labels, seq_lengths, padding_mask) = batch
    
            event_ids = batch[8]  # Ensure index matches your unpacking
            predictions, attn_weights = model_instance(*inputs, seq_lengths)

            outcome_for_mask = labels.view(-1) if target_task.task == 'classification' else labels
            valid_mask = compute_event_mask(event_ids, outcome_for_mask, valid_event_ids, device)
            #print(valid_mask)
            if valid_mask.sum() == 0:
                continue
            optimizer.zero_grad()
            if target_task.task == 'regression':
                 # Regression: apply mask to flattened views to stay consistent
                loss = criterion(predictions.view(-1), labels.view(-1))[valid_mask.view(-1)].mean()

            else:
                
                predictions_flat = predictions.view(-1, target_task.num_labels)
                loss = criterion(predictions_flat[valid_mask], labels.view(-1)[valid_mask]).mean()
                        
            loss.backward()
            # if "grad_clip" in optim_cfg:
            #     torch.nn.utils.clip_grad_norm_(model_instance.parameters(), optim_cfg["grad_clip"])
            optimizer.step()
            total_train_loss += loss.item()
            n_trained_batches += 1  # ← only count batches that trained


        # --- Validation ---
        metrics, _, _, _ = validation(model_instance, target_task, val_loader, event2id,
                                       valid_event_ids, device=device)

        val_batch_loss = metrics['avg_batch_loss']
        val_note_loss = metrics['avg_note_loss']
        val_note_score = metrics['avg_note_SCORE']
        val_note_auc = metrics['avg_note_AUC']

        # after epoch
        avg_train_loss = total_train_loss / max(n_trained_batches, 1)  # ← avoid div by zero
      
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | "
            f"Val Loss={val_note_loss:.4f} | "
            f"Val BA={val_note_score:.4f} | "
            f"Val AUC={val_note_auc:.4f}")
        val_note_score = val_note_auc
        scheduler.step(val_note_score)   

        if val_note_score > best_val_score:
            best_val_score = val_note_score
            best_model_state = copy.deepcopy(model_instance.state_dict())
            patience_counter = 0
            print(f"  ✓ New best ROC-AUC: {val_note_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return best_val_score, best_model_state


def eval_model(model, event2id, target_task, config, val_loader, df_val, split, valid_event_ids, target_event_types, device='cuda'):
    # --- Validation evaluation ---
    metrics_val, all_preds, all_labels, all_probs = validation(
        model, target_task, val_loader, event2id, valid_event_ids, device=device
    )
    print(
        f"Final test scores: Global Batch Loss = {metrics_val['avg_batch_loss']:.4f}, Global Loss = {metrics_val['avg_note_loss']:.4f}, "
        f"Global BA = {metrics_val['avg_note_SCORE']:.4f}, Global AUC = {metrics_val['avg_note_AUC']:.4f}")

    # --- Use Evaluator to store predictions & metrics ---
    evaluator = Evaluator()
    split_scores = {
        split: (all_preds, all_probs, all_labels, df_val)
    }

    evaluator.add_results(
        learner="multimodal_model",
        input_config=config,
        split_scores=split_scores,
        task=target_task,
        event_types=target_event_types
    )

    return evaluator.results, evaluator.predictions


def flatten_logits(logits, target_task):
    # flatten the predictions, labels and mask for classification.
    logits = logits.view(-1, target_task.num_labels)  # [batch*seq_len, num_classes]
    return logits


def validation(model, target_task, val_loader, event2id, valid_event_ids=None, device='cuda'):
    """
    Evaluate the model on a validation/test set.

    Returns:
        metrics : dict
            Aggregated metrics including avg_batch_loss, avg_note_loss, avg_score, and optionally AUC.
        all_predictions : np.ndarray
            Concatenated predicted labels or values.
        all_labels : np.ndarray
            Concatenated true labels.
        all_probs : np.ndarray
            Predicted probabilities for classification (None for regression).
    """
    model.eval()
    total_batch_loss, total_note_loss, total_valid_elements = 0, 0, 0
    all_predictions, all_labels, all_probs = [], [], []
    # Define loss
    criterion = nn.L1Loss(reduction='none') if target_task.task == 'regression' \
        else nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    with torch.no_grad():
        for batch in val_loader:
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            (*inputs, labels, seq_lengths, padding_mask) = batch
           
            event_ids = batch[8]  # Ensure index matches your unpacking
            logits, attn_weights = model(*inputs, seq_lengths)

            outcome_for_mask = labels.view(-1) if target_task.task == 'classification' else labels
            valid_mask = compute_event_mask(event_ids, outcome_for_mask, valid_event_ids, device)

            if valid_mask.sum() == 0:
                continue

            if target_task.task == 'classification':
                logits_flat = logits.view(-1, target_task.num_labels)
                labels_flat = labels.view(-1)

                per_token_loss = criterion(logits_flat, labels_flat)
                loss = per_token_loss[valid_mask].mean()

                all_predictions.append(logits_flat.argmax(dim=-1)[valid_mask])
                all_labels.append(labels_flat[valid_mask])
                all_probs.append(torch.softmax(logits_flat, dim=-1)[valid_mask])

            else:  # regression
                per_token_loss = criterion(logits.view(-1), labels.view(-1))
                loss = per_token_loss[valid_mask.view(-1)].mean()

                all_predictions.append(logits.view(-1)[valid_mask.view(-1)])
                all_labels.append(labels.view(-1)[valid_mask.view(-1)])

            total_batch_loss += loss.item()
            total_note_loss += per_token_loss[valid_mask.view(-1)].sum().item()
            total_valid_elements += valid_mask.sum().item()

    # ── Aggregate across full validation set ──────────────────────────────────
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy() if all_probs else None

    avg_batch_loss = total_batch_loss / len(val_loader)
    avg_note_loss = total_note_loss / max(total_valid_elements, 1)

    # ── Metrics computed ONCE on full set — no single-class warnings ──────────
    note_score = target_task.evaluate(all_labels, all_predictions)
    note_auc = None
    if all_probs is not None and hasattr(target_task, "evaluate_prob"):
        note_auc = target_task.evaluate_prob(all_labels, all_probs)

    metrics = {
        "avg_batch_loss": avg_batch_loss,
        "avg_note_loss": avg_note_loss,
        "avg_note_SCORE": note_score,
        "avg_note_AUC": note_auc,
    }
    return metrics, all_predictions, all_labels, all_probs
