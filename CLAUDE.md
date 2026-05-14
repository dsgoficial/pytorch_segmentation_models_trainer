# OpenWolf

@.wolf/OPENWOLF.md

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md every session. Check .wolf/cerebrum.md before generating code. Check .wolf/anatomy.md before reading files.


# Project guidelines for Claude Code

## Ao implementar qualquer feature

Toda implementação deve incluir os itens abaixo antes de ser considerada concluída.
Nós desenvolvemos usando TDD sempre.
Novas funcionalidades devem ter 100% de coverage nos testes desenvolvidos.
Os commits não devem nunca diminuir a coverage do projeto. Toda modificação deve olhar para os testes daquela componente e garantir que a coverage dele não será diminuída.
Não espere o usuário pedir cada um separadamente:

1. **Testes unitários** em `tests/test_<módulo>.py` cobrindo:
   - Contrato de retorno (tipo, shape, dtype)
   - Casos de borda (batch sizes diferentes, entradas degeneradas)
   - Fluxo de gradiente quando relevante
   - Integração com o LightningModule proprietário

2. **Documentação de código** (docstrings) em toda classe e método público novo,
   incluindo Args, Returns e um exemplo YAML quando aplicável.

3. **Documentação de usuário** em `website/docs/` (Docusaurus Markdown):
   - Arquivo novo para features maiores com `sidebar_position` e `title` no frontmatter
   - Atualização dos arquivos existentes para linkar a nova feature
   - Exemplos de config YAML funcionais

4. **Atualização do `CHANGELOG.md`** (regra detalhada abaixo).

5. **Exemplo de config YAML** em `conf/examples/` para toda feature configurável via Hydra.

---

## Changelog

**Sempre atualizar `CHANGELOG.md` ao modificar o framework.**

- Todo novo feature, bug fix, refatoração ou adição de documentação deve ter uma entrada correspondente em `# Unreleased` antes de a tarefa ser considerada concluída.
- Agrupar entradas em subsections `##` descritivas (ex: `## Domain Adaptation`, `## Bug fixes`).
- Cada bullet deve ser específico o suficiente para o usuário entender o que mudou e por quê, sem precisar ler o diff.
- Não criar nova seção versionada — isso é feito no momento do release. Todo trabalho em andamento vai em `# Unreleased`.

---

## Testes

- Ambiente: Gerenciado pelo `uv`.
- Comando padrão: `uv run pytest tests/ -v --tb=short`
- Ignorar por padrão (lentos / rede / segfault conhecido):
  - `tests/test_detection_model.py` — segfault conhecido
  - `tests/test_inference.py` — baixa modelos da rede
  - `tests/test_predict.py` — baixa checkpoints
  - `tests/test_script.py` — spawna subprocessos

---

## Padrões arquiteturais

- Separar sempre `nn.Module` (lógica de domínio) de `pl.LightningModule` (orquestração do treino).
  Exemplo canônico: `BaseDomainAdaptationMethod` (nn.Module) + `DomainAdaptationModel` (LightningModule).
- Toda classe nova instanciável via Hydra deve aceitar `**kwargs` no `__init__` e ser testável
  sem Trainer real (usar `MagicMock` para o trainer nos testes).
- Configurações novas devem ter um dataclass correspondente em `config_definitions/`
  registrado no ConfigStore.
