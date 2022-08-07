import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

QSAROME_34_cutoff = {
    '34_targets_curated_data_NORM_5-HT1A.SDF': 7,
    '34_targets_curated_data_NORM_5-HT1B.SDF': 7,
    '34_targets_curated_data_NORM_5-HT1D.SDF': 7,
    '34_targets_curated_data_NORM_5-HT1E.SDF': 5,
    '34_targets_curated_data_NORM_5-HT2A.SDF': 7,
    '34_targets_curated_data_NORM_5-HT2C.SDF': 7,
    '34_targets_curated_data_NORM_5-HT3.SDF': 5,
    '34_targets_curated_data_NORM_5-HT5.SDF': 6,
    '34_targets_curated_data_NORM_5-HT6.SDF': 7,
    '34_targets_curated_data_NORM_5-HT7.SDF': 7,
    '34_targets_curated_data_NORM_Alpha-1A.SDF': 7,
    '34_targets_curated_data_NORM_Alpha-1B.SDF': 7,
    '34_targets_curated_data_NORM_Alpha-2A.SDF': 7,
    '34_targets_curated_data_NORM_Alpha-2B.SDF': 7,
    '34_targets_curated_data_NORM_Alpha-2C.SDF': 7,
    '34_targets_curated_data_NORM_Beta-1.SDF': 7,
    '34_targets_curated_data_NORM_Beta-2.SDF': 7,
    '34_targets_curated_data_NORM_DOPAMINE D1.SDF': 7,
    '34_targets_curated_data_NORM_DOPAMINE D2.SDF': 7,
    '34_targets_curated_data_NORM_DOPAMINE D3.SDF': 7,
    '34_targets_curated_data_NORM_DOPAMINE D4.SDF': 7,
    '34_targets_curated_data_NORM_DOPAMINE D5.SDF': 6,
    '34_targets_curated_data_NORM_Dopamine transporter.SDF': 7,
    '34_targets_curated_data_NORM_HISTAMINE H1.SDF': 7,
    '34_targets_curated_data_NORM_HISTAMINE H2.SDF': 5,
    '34_targets_curated_data_NORM_HISTAMINE H3.SDF': 8,
    '34_targets_curated_data_NORM_HISTAMINE H4.SDF': 7,
    '34_targets_curated_data_NORM_Muscarinic acetylcholine receptor M1.SDF': 7,
    '34_targets_curated_data_NORM_Muscarinic acetylcholine receptor M2.SDF': 7,
    '34_targets_curated_data_NORM_Muscarinic acetylcholine receptor M3.SDF': 7,
    '34_targets_curated_data_NORM_Muscarinic acetylcholine receptor M4.SDF': 6,
    '34_targets_curated_data_NORM_Muscarinic acetylcholine receptor M5.SDF': 6,
    '34_targets_curated_data_NORM_Norepinephrine transporter.SDF': 7,
    '34_targets_curated_data_NORM_Serotonin transporter.SDF': 7
}