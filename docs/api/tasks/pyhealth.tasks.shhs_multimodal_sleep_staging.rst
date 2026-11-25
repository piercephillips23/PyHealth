Sleep staging that builds on ``NSRRSleepDataset`` and the existing
cardiorespiratory sleep task. It accepts SHHS EDF and NSRR scored-event or
profusion XML files. All samples have ECG (1024 points), ABD and THX (256
points each), a four-class label, and an availability mask in ECG/ABD/THX
order. Use participant-level splits. The default is offline whole-recording
normalization and polyphase resampling; the task does not model overnight
context. Naturally absent or unusable signals are zero-filled. When selecting
modalities in the task constructor, unselected signals are also unavailable to
the resulting task. For comparing fixed subsets, prefer the default task and
apply the evaluation masking utility to the batch.

The masking policies are dynamic per epoch (not per night), and the paper
policy conditions independent drops on a nonempty subset. Balanced masking
samples uniformly from nonempty subsets. Neither policy mutates cached data.
The reference is Carter and Tarassenko (2024),
`wav2sleep <https://arxiv.org/abs/2411.04644>`.

SHHS multimodal sleep staging
=============================

.. autoclass:: pyhealth.tasks.shhs_multimodal_sleep_staging.SHHSMultimodalSleepStaging
   :members:
   :undoc-members:
   :show-inheritance:

Masking utilities
-----------------

.. automodule:: pyhealth.tasks.shhs_multimodal_masking
   :members: sample_input_mask, collate_shhs_epochs, apply_modality_mask
   :undoc-members:
   :show-inheritance:
