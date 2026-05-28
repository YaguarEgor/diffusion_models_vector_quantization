# Diffusion Quantization Research Map

Рабочая карта related work и направлений для курсовой.

## Главная рамка

Текущую работу лучше позиционировать как training-free RVQ-подход к квантизации весов diffusion-модели:

- residual vector quantization вместо дорогого additive/beam-search VQ;
- stage-wise распределение кодового бюджета через разные `K` по стадиям;
- опциональное ортогональное преобразование перед квантизацией;
- full-XTX post-RVQ recoding: codebooks фиксируются, а assignments дооптимизируются под полный XTX-функционал;
- diffusion-aware calibration через prompts/seeds/timesteps/activation statistics.

## 1. Возможные бейзлайны

Эти работы нужны, чтобы понять, с кем сравниваться по качеству, bpp и сложности запуска.

### Q-Diffusion

Классический diffusion-specific PTQ baseline.

Зачем нужен:
- показать ранний сильный baseline для квантизации diffusion models;
- использовать как отправную точку для timestep-aware calibration и PTQ-методологии.

### BitsFusion

Сильный low-bit baseline для Stable Diffusion, особенно важен из-за SD1.5 UNet.

Зачем нужен:
- понять, насколько сильны современные scalar/group low-bit методы;
- сравнить целевой диапазон около 2 bits;
- оценить, кого реально можно попытаться догнать в первую очередь.

### EDA-DM / PTQD / TFMQ-DM

Diffusion-aware PTQ методы, связанные с calibration, timesteps и distribution alignment.

Зачем нужны:
- обосновать, почему сбор `XTX` по правильным prompts/seeds/timesteps важен;
- сравнить подходы к calibration distribution;
- взять идеи для улучшения нашего `XTX`.

### VQDM

Главный conceptual vector-quantization baseline, но вычислительно тяжелый.

Зачем нужен:
- показать связь с additive/vector quantization;
- объяснить, почему мы ищем более дешевую training-free RVQ-альтернативу;
- сравнивать аккуратно, потому что VQDM использует дорогой beam search и finetuning.

### AQUATIC-Diff

Additive/codebook-based quantization для compressed diffusion models.

Зачем нужен:
- очень близкая работа по идее codebook-based/additive quantization;
- полезен как ближайший related work к RVQ/VQ-ветке, даже если модель/setting отличаются от SD1.5.

## 2. Ортогональное преобразование

### TR-DQ

Time-Rotation Diffusion Quantization.

Зачем нужен:
- обосновывает идею rotation/Hadamard как неслучайную эвристику;
- показывает, что поворот пространства может сгладить веса/активации перед low-bit quantization;
- дает язык для описания нашего Hadamard: fixed orthogonal transform as a cheap rotation preconditioner.

Текущий вывод по нашим запускам:
- Hadamard помогает не всегда;
- его стоит оставлять как ablation, но не продавать как главный источник выигрыша;
- если развивать дальше, логичный шаг - обучаемая или data-aware rotation.

## 3. Recode / reassignment

### VQ4DiT

Post-training vector quantization для Diffusion Transformers.

Зачем нужен:
- хорошая мотивация для того, что в VQ важны не только codebooks, но и assignments;
- напрямую поддерживает идею post-RVQ recoding/refinement;
- близок к нашей логике: codebooks уже обучены, дальше улучшаем codes.

Текущий вывод по нашим запускам:
- final full-XTX recoding выглядит полезным и должен оставаться включенным;
- local recode после каждой RVQ-стадии пока выглядит вредным: он слишком рано оптимизирует частичный residual и мешает следующим стадиям.

## 4. Сбор активаций и calibration

### EDA-DM

Enhanced Distribution Alignment для PTQ diffusion models.

Зачем нужен:
- обосновать, что calibration distribution в diffusion-моделях особенно важен;
- связать качество `XTX` с выбором prompts, timesteps, seeds и activation statistics;
- объяснить, почему один слой или маленький набор prompts может быть нечестной оценкой.

Идеи для развития:
- timestep-weighted `XTX`;
- больше calibration prompts на низких bpp;
- разные calibration profiles для attention/FFN/conv;
- проверка чувствительности к prompts/seeds.

## 5. Возможное улучшение через outliers

### DMQ

Dissecting outliers of diffusion models for PTQ.

Зачем нужен:
- показывает, что outliers в diffusion-моделях могут сильно ломать low-bit quantization;
- дает направление для улучшения RVQ без перехода к полному VQDM.

Возможные улучшения нашего метода:
- channel scaling перед RVQ;
- outlier-aware Hadamard/rotation;
- отдельная обработка тяжелых каналов;
- mixed precision для чувствительных слоев;
- timestep-weighted или outlier-weighted XTX.

### DGQ

Distribution-aware group quantization для text-to-image diffusion models.

Текущий статус:
- не включать в основные бейзлайны: полный DGQ требует attention-score/alignment quantization, а наша DGQ-lite реализация была только group/outlier-aware PTQ приближением;
- использовать как источник идей, а не как прямой baseline.

Что можно взять для развития:
- outlier-aware обработку активаций;
- prompt/timestep-aware calibration;
- отдельную работу с cross-attention, где качество нужно проверять через `clip_ti` и `delta_clip_ti`;
- возможную post-softmax attention probability quantization как лёгкую исследовательскую ветку поверх RVQ.

## 6. Практический план сравнения

Для каждого baseline стоит вытащить:

- модель и dataset/eval setting;
- какие части модели квантуются: UNet, text encoder, VAE, activations;
- weight bpp / activation bits;
- FID, CLIP score, ImageReward/HPS/PickScore, LPIPS, если есть;
- есть ли finetuning/QAT или это чистый PTQ;
- работает ли метод на SD1.5 text-to-image или на более простом diffusion setting;
- насколько честно сравнивать с нашим текущим setup.

Приоритет для ближайшего анализа:

1. BitsFusion: вероятно, самый важный SD1.5 low-bit baseline.
2. Q-Diffusion: классический reference point.
3. PTQ4DM / старый scalar PTQ: простые PTQ-бейзлайны для нижней границы качества.
4. AQUATIC-Diff: ближайшая codebook/additive ветка.
5. VQ4DiT: аргумент для recoding, не обязательно прямой baseline.
6. DGQ, TR-DQ и DMQ: идеи для улучшения, не основные baselines.

## 7. Будущая learned / distillation ветка

Текущую экспериментальную рамку важно отделять от обучаемых вариантов:

- текущий RVQ и текущие baseline-файлы лучше считать `post-training / calibration-only` сравнением;
- корректнее формулировать не просто `training-free`, а `без fine-tuning diffusion model`;
- у RVQ кодбуки и assignments подбираются, но сама SD1.5 не дообучается по denoising loss;
- у BitsFusion/QDiff/PTQ4DM-lite мы тоже не делаем QAT, distillation и обновление параметров UNet.

Почему это важно:
- если добавить QAT/distillation в BitsFusion-like baseline, он становится уже другим классом метода;
- тогда сравнивать его напрямую с текущим RVQ будет не совсем честно;
- зато это открывает честную будущую ветку: learned/distilled quantization methods.

Возможное развитие BitsFusion-like:
- оставить full `mid_block + conv`;
- добавить fake-quant модули, learnable scales/clipping и/или bit logits;
- учить только quantization parameters, а не полный UNet;
- использовать FP SD1.5 как teacher;
- objective: teacher-student denoising consistency, например
  `||eps_teacher(x_t, t, c) - eps_student(x_t, t, c)||^2`;
- отдельно учитывать bpp constraint через penalty/Lagrangian.

Возможное развитие нашего RVQ:
- заморозить FP SD1.5 как teacher;
- использовать RVQ-квантованную модель как student;
- не менять исходные FP-веса напрямую;
- доучивать только компактные параметры квантизации:
  codebook vectors, per-stage/per-layer scales, rotation parameters, maybe small correction/adapters;
- assignments можно либо фиксировать, либо обновлять редко, либо учить через STE/soft assignment;
- objective тот же: teacher-student denoising loss по prompts/seeds/timesteps.

Наиболее перспективная формулировка:

> RVQ gives a strong post-training initialization for later quantization-aware distillation.

То есть текущая курсовая может показывать, что calibration-only RVQ уже конкурентен, а будущая работа - что RVQ является хорошей стартовой точкой для learned quantization-aware distillation.
