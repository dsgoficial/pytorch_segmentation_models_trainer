import { useState } from 'react'
import ModelSection from '../components/ModelSection'
import ImportModal from '../components/ImportModal'
import YamlPreview from '../components/YamlPreview'

// ── Estado inicial ─────────────────────────────────────────────────────────────

const INITIAL_MODEL = {
  architecture: 'segmentation_models_pytorch.Unet',
  encoderName: 'resnet101',
  encoderWeights: 'None',
  inChannels: 12,
  classes: 7,
}

const INITIAL_HP = { numClasses: 7, batchSize: 100 }

const INITIAL_TRAINER = { accelerator: 'gpu', devices: 1, precision: 16 }

const INITIAL_PROCESSOR = {
  target: 'pytorch_segmentation_models_trainer.inference.inference_processors.MultiClassInferenceProcessor',
  modelInputH: 256,
  modelInputW: 256,
  stepH: 128,
  stepW: 128,
  useNormalize: false,
  normalizeMean: '',
  normalizeStd: '',
}

const INITIAL_IMAGE_READER = {
  folderName: '',
  recursive: false,
  imageExtension: '.tif',
}

const INITIAL_EXPORT = {
  outputFilePath: '',
}

const PROCESSOR_TYPES = [
  { value: 'pytorch_segmentation_models_trainer.inference.inference_processors.MultiClassInferenceProcessor',  label: 'MultiClassInferenceProcessor' },
  { value: 'pytorch_segmentation_models_trainer.inference.inference_processors.SingleImageInferenceProcessor', label: 'SingleImageInferenceProcessor' },
]

// ── Serialização ───────────────────────────────────────────────────────────────

function parseCsvNumbers(str) {
  return str.split(',').map(s => Number(s.trim())).filter(n => !isNaN(n))
}

function buildConfig(checkpointPath, device, saveInference, model, hp, trainer, processor, imageReader, exportStrategy) {
  const normMean = processor.useNormalize ? parseCsvNumbers(processor.normalizeMean) : null
  const normStd  = processor.useNormalize ? parseCsvNumbers(processor.normalizeStd)  : null

  return {
    checkpoint_path: checkpointPath,
    pl_model: { _target_: 'pytorch_segmentation_models_trainer.model_loader.model.Model' },
    device,
    hyperparameters: {
      num_classes: hp.numClasses,
      batch_size: hp.batchSize,
    },
    save_inference: saveInference,
    pl_trainer: {
      precision: trainer.precision,
      devices: trainer.devices,
      accelerator: trainer.accelerator,
    },
    model: {
      _target_: model.architecture,
      encoder_name: model.encoderName,
      ...(model.encoderWeights !== 'None' && { encoder_weights: model.encoderWeights }),
      in_channels: model.inChannels,
      classes: '${hyperparameters.num_classes}',
    },
    inference_processor: {
      _target_: processor.target,
      model_input_shape: [processor.modelInputH, processor.modelInputW],
      step_shape: [processor.stepH, processor.stepW],
      num_classes: '${hyperparameters.num_classes}',
      normalize_mean: normMean,
      normalize_std: normStd,
    },
    inference_image_reader: {
      _target_: 'pytorch_segmentation_models_trainer.inference.image_reader.FolderImageReaderProcessor',
      folder_name: imageReader.folderName,
      recursive: imageReader.recursive,
      image_extension: imageReader.imageExtension,
    },
    export_strategy: {
      _target_: 'pytorch_segmentation_models_trainer.inference.export.RasterExportInferenceStrategy',
      output_file_path: exportStrategy.outputFilePath,
    },
  }
}

function applyImport(data, setters) {
  const { setCheckpointPath, setDevice, setSaveInference, setModel, setHp,
          setTrainer, setProcessor, setImageReader, setExportStrategy } = setters

  if (data.checkpoint_path != null) setCheckpointPath(String(data.checkpoint_path))
  if (data.device != null) setDevice(data.device)
  if (data.save_inference != null) setSaveInference(Boolean(data.save_inference))

  if (data.model) {
    setModel({
      architecture:   data.model._target_,
      encoderName:    data.model.encoder_name,
      encoderWeights: data.model.encoder_weights ?? 'None',
      inChannels:     data.model.in_channels,
      classes:        data.model.classes,
    })
  }
  if (data.hyperparameters) {
    setHp({
      numClasses: data.hyperparameters.num_classes ?? 7,
      batchSize:  data.hyperparameters.batch_size  ?? 100,
    })
  }
  if (data.pl_trainer) {
    setTrainer({
      accelerator: data.pl_trainer.accelerator ?? 'gpu',
      devices:     data.pl_trainer.devices     ?? 1,
      precision:   data.pl_trainer.precision   ?? 16,
    })
  }
  if (data.inference_processor) {
    const ip = data.inference_processor
    const mean = Array.isArray(ip.normalize_mean) ? ip.normalize_mean.join(', ') : ''
    const std  = Array.isArray(ip.normalize_std)  ? ip.normalize_std.join(', ')  : ''
    setProcessor({
      target:         ip._target_,
      modelInputH:    ip.model_input_shape?.[0] ?? 256,
      modelInputW:    ip.model_input_shape?.[1] ?? 256,
      stepH:          ip.step_shape?.[0] ?? 128,
      stepW:          ip.step_shape?.[1] ?? 128,
      useNormalize:  ip.normalize_mean != null,
      normalizeMean: mean,
      normalizeStd:  std,
    })
  }
  if (data.inference_image_reader) {
    const ir = data.inference_image_reader
    setImageReader({
      folderName:     ir.folder_name      ?? '',
      recursive:      ir.recursive        ?? false,
      imageExtension: ir.image_extension  ?? '.tif',
    })
  }
  if (data.export_strategy) {
    setExportStrategy({ outputFilePath: data.export_strategy.output_file_path ?? '' })
  }
}

// ── Componente principal ───────────────────────────────────────────────────────

export default function PredictPage() {
  const [checkpointPath, setCheckpointPath] = useState('')
  const [device, setDevice]                 = useState('cuda')
  const [saveInference, setSaveInference]   = useState(true)
  const [model, setModel]                   = useState(INITIAL_MODEL)
  const [hp, setHp]                         = useState(INITIAL_HP)
  const [trainer, setTrainer]               = useState(INITIAL_TRAINER)
  const [processor, setProcessor]           = useState(INITIAL_PROCESSOR)
  const [imageReader, setImageReader]       = useState(INITIAL_IMAGE_READER)
  const [exportStrategy, setExportStrategy] = useState(INITIAL_EXPORT)
  const [showImport, setShowImport]         = useState(false)

  function handleImport(data) {
    applyImport(data, {
      setCheckpointPath, setDevice, setSaveInference, setModel, setHp,
      setTrainer, setProcessor, setImageReader, setExportStrategy,
    })
  }

  const config = buildConfig(checkpointPath, device, saveInference, model, hp, trainer, processor, imageReader, exportStrategy)

  return (
    <>
      <div style={styles.toolbar}>
        <button onClick={() => setShowImport(true)} style={styles.importBtn}>
          Import YAML
        </button>
      </div>

      {showImport && (
        <ImportModal onImport={handleImport} onClose={() => setShowImport(false)} />
      )}

      <main style={styles.main}>
        <div style={styles.formCol}>

          {/* ── Checkpoint & Device ─────────────────────────────────────────── */}
          <section style={s.section}>
            <h2>Checkpoint &amp; Device</h2>
            <div className="field">
              <label>checkpoint_path</label>
              <input
                type="text"
                value={checkpointPath}
                onChange={e => setCheckpointPath(e.target.value)}
                placeholder="/path/to/model.ckpt"
              />
            </div>
            <div style={s.row}>
              <div className="field" style={{ flex: 1 }}>
                <label>device</label>
                <select value={device} onChange={e => setDevice(e.target.value)}>
                  <option value="cuda">cuda</option>
                  <option value="cpu">cpu</option>
                </select>
              </div>
              <div className="field" style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, paddingTop: 20 }}>
                <input
                  type="checkbox"
                  id="save_inference"
                  checked={saveInference}
                  onChange={e => setSaveInference(e.target.checked)}
                  style={{ width: 'auto' }}
                />
                <label htmlFor="save_inference" style={{ margin: 0 }}>save_inference</label>
              </div>
            </div>
          </section>

          {/* ── Hyperparameters ─────────────────────────────────────────────── */}
          <section style={s.section}>
            <h2>Hyperparameters</h2>
            <div style={s.row}>
              <div className="field" style={{ flex: 1 }}>
                <label>num_classes</label>
                <input
                  type="number"
                  min={1}
                  value={hp.numClasses}
                  onChange={e => setHp(p => ({ ...p, numClasses: Number(e.target.value) }))}
                />
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label>batch_size</label>
                <input
                  type="number"
                  min={1}
                  value={hp.batchSize}
                  onChange={e => setHp(p => ({ ...p, batchSize: Number(e.target.value) }))}
                />
              </div>
            </div>
          </section>

          {/* ── PL Trainer ──────────────────────────────────────────────────── */}
          <section style={s.section}>
            <h2>PL Trainer</h2>
            <div style={s.row}>
              <div className="field" style={{ flex: 1 }}>
                <label>Accelerator</label>
                <select value={trainer.accelerator} onChange={e => setTrainer(p => ({ ...p, accelerator: e.target.value }))}>
                  <option value="gpu">gpu</option>
                  <option value="cpu">cpu</option>
                  <option value="auto">auto</option>
                </select>
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label>Devices</label>
                <input
                  type="number"
                  min={1}
                  value={trainer.devices}
                  onChange={e => setTrainer(p => ({ ...p, devices: Number(e.target.value) }))}
                />
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label>Precision</label>
                <select value={trainer.precision} onChange={e => setTrainer(p => ({ ...p, precision: e.target.value }))}>
                  <option value={32}>32 (float32)</option>
                  <option value={16}>16 (float16 / AMP)</option>
                  <option value="bf16">bf16</option>
                  <option value="16-mixed">16-mixed</option>
                </select>
              </div>
            </div>
          </section>

          {/* ── Model ───────────────────────────────────────────────────────── */}
          <ModelSection model={model} onChange={setModel} />

          {/* ── Inference Processor ─────────────────────────────────────────── */}
          <section style={s.section}>
            <h2>Inference Processor</h2>

            <div className="field">
              <label>Type (_target_)</label>
              <select
                value={processor.target}
                onChange={e => setProcessor(p => ({ ...p, target: e.target.value }))}
              >
                {PROCESSOR_TYPES.map(t => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </div>

            <div style={s.row}>
              <div className="field" style={{ flex: 1 }}>
                <label>model_input_shape H</label>
                <input
                  type="number" min={1}
                  value={processor.modelInputH}
                  onChange={e => setProcessor(p => ({ ...p, modelInputH: Number(e.target.value) }))}
                />
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label>model_input_shape W</label>
                <input
                  type="number" min={1}
                  value={processor.modelInputW}
                  onChange={e => setProcessor(p => ({ ...p, modelInputW: Number(e.target.value) }))}
                />
              </div>
            </div>

            <div style={s.row}>
              <div className="field" style={{ flex: 1 }}>
                <label>step_shape H</label>
                <input
                  type="number" min={1}
                  value={processor.stepH}
                  onChange={e => setProcessor(p => ({ ...p, stepH: Number(e.target.value) }))}
                />
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label>step_shape W</label>
                <input
                  type="number" min={1}
                  value={processor.stepW}
                  onChange={e => setProcessor(p => ({ ...p, stepW: Number(e.target.value) }))}
                />
              </div>
            </div>

            <div className="field" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                type="checkbox"
                id="use_normalize"
                checked={processor.useNormalize}
                onChange={e => setProcessor(p => ({ ...p, useNormalize: e.target.checked }))}
                style={{ width: 'auto' }}
              />
              <label htmlFor="use_normalize" style={{ margin: 0 }}>
                Usar normalize_mean / normalize_std
              </label>
            </div>

            {processor.useNormalize && (
              <>
                <div className="field">
                  <label>normalize_mean <span style={s.hint}>(valores separados por vírgula)</span></label>
                  <input
                    type="text"
                    value={processor.normalizeMean}
                    onChange={e => setProcessor(p => ({ ...p, normalizeMean: e.target.value }))}
                    placeholder="70.5, 77.6, 40.9, ..."
                  />
                </div>
                <div className="field">
                  <label>normalize_std <span style={s.hint}>(valores separados por vírgula)</span></label>
                  <input
                    type="text"
                    value={processor.normalizeStd}
                    onChange={e => setProcessor(p => ({ ...p, normalizeStd: e.target.value }))}
                    placeholder="27.0, 20.1, 18.0, ..."
                  />
                </div>
              </>
            )}
          </section>

          {/* ── Image Reader ────────────────────────────────────────────────── */}
          <section style={s.section}>
            <h2>Image Reader</h2>
            <div className="field">
              <label>folder_name</label>
              <input
                type="text"
                value={imageReader.folderName}
                onChange={e => setImageReader(p => ({ ...p, folderName: e.target.value }))}
                placeholder="/path/to/images"
              />
            </div>
            <div style={s.row}>
              <div className="field" style={{ flex: 1 }}>
                <label>image_extension</label>
                <input
                  type="text"
                  value={imageReader.imageExtension}
                  onChange={e => setImageReader(p => ({ ...p, imageExtension: e.target.value }))}
                  placeholder=".tif"
                />
              </div>
              <div className="field" style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, paddingTop: 20 }}>
                <input
                  type="checkbox"
                  id="recursive"
                  checked={imageReader.recursive}
                  onChange={e => setImageReader(p => ({ ...p, recursive: e.target.checked }))}
                  style={{ width: 'auto' }}
                />
                <label htmlFor="recursive" style={{ margin: 0 }}>recursive</label>
              </div>
            </div>
          </section>

          {/* ── Export Strategy ─────────────────────────────────────────────── */}
          <section style={s.section}>
            <h2>Export Strategy</h2>
            <div className="field">
              <label>output_file_path</label>
              <input
                type="text"
                value={exportStrategy.outputFilePath}
                onChange={e => setExportStrategy({ outputFilePath: e.target.value })}
                placeholder="/path/to/output"
              />
            </div>
          </section>

        </div>
        <div style={styles.previewCol}>
          <YamlPreview config={config} />
        </div>
      </main>
    </>
  )
}

const s = {
  section: {
    background: '#fff',
    border: '1px solid #e5e5e5',
    borderRadius: 8,
    padding: 20,
    marginBottom: 16,
  },
  row: { display: 'flex', gap: 12 },
  hint: { fontSize: '0.72rem', color: '#aaa' },
}

const styles = {
  toolbar:    { padding: '8px 24px', borderBottom: '1px solid #e5e5e5', display: 'flex', justifyContent: 'flex-end', background: '#fff' },
  importBtn:  { background: 'none', border: '1px solid #3b82f6', color: '#3b82f6', borderRadius: 6, padding: '6px 14px', fontSize: '0.85rem', cursor: 'pointer', fontWeight: 500 },
  main:       { display: 'grid', gridTemplateColumns: '1fr 1fr', flex: 1, minHeight: 0 },
  formCol:    { padding: 24, overflowY: 'auto', borderRight: '1px solid #e5e5e5', background: '#f5f5f5' },
  previewCol: { padding: 0, background: '#1e1e1e', overflow: 'hidden' },
}
