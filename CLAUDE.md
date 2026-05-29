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

  ---

  ## Karpathy Guidelines

  Behavioral guidelines to reduce common LLM coding mistakes, derived from Andrej Karpathy's observations on LLM coding pitfalls.

  Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

  ### 1. Think Before Coding
  Don't assume. Don't hide confusion. Surface tradeoffs.
  Before implementing:
  - State your assumptions explicitly. If uncertain, ask.
  - If multiple interpretations exist, present them - don't pick silently.
  - If a simpler approach exists, say so. Push back when warranted.
  - If something is unclear, stop. Name what's confusing. Ask.

  ### 2. Simplicity First
  Minimum code that solves the problem. Nothing speculative.
  - No features beyond what was asked.
  - No abstractions for single-use code.
  - No "flexibility" or "configurability" that wasn't requested.
  - No error handling for impossible scenarios.
  - If you write 200 lines and it could be 50, rewrite it.
  - Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

  ### 3. Surgical Changes
  Touch only what you must. Clean up only your own mess.
  When editing existing code:
  - Don't "improve" adjacent code, comments, or formatting.
  - Don't refactor things that aren't broken.
  - Match existing style, even if you'd do it differently.
  - If you notice unrelated dead code, mention it - don't delete it.
  When your changes create orphans:
  - Remove imports/variables/functions that YOUR changes made unused.
  - Don't remove pre-existing dead code unless asked.
  *The test:* Every changed line should trace directly to the user's request.

  ### 4. Goal-Driven Execution
  Define success criteria. Loop until verified.
  Transform tasks into verifiable goals:
  - "Add validation" → "Write tests for invalid inputs, then make them pass"
  - "Fix the bug" → "Write a test that reproduces it, then make it pass"
  - "Refactor X" → "Ensure tests pass before and after"
  For multi-step tasks, state a brief plan:
  1. [Step] → verify: [check]
  2. [Step] → verify: [check]
  3. [Step] → verify: [check]
  Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
