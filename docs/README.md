# Documentation

This folder contains the documentation for PyTorch Segmentation Models Trainer.

## Building Documentation

### Prerequisites

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-jupyter
```

### Local Development

```bash
# Serve documentation locally with auto-reload
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

### Building Static Site

```bash
# Build static HTML files
mkdocs build

# Output will be in site/ directory
```

## Deployment

### GitHub Pages

```bash
# Deploy to GitHub Pages (requires push access)
mkdocs gh-deploy
```

### Manual Deployment

1. Build the documentation:
   ```bash
   mkdocs build
   ```

2. Copy the `site/` folder contents to your web server.

## Writing Documentation

### File Structure

- `docs/` - All documentation files
- `mkdocs.yml` - Configuration file
- Use `.md` files for content
- Follow the navigation structure in `mkdocs.yml`

### Writing Guidelines

1. **Use descriptive headings** with proper hierarchy
2. **Include code examples** with syntax highlighting
3. **Add tips and warnings** using admonitions:
   ```markdown
   !!! tip "Pro Tip"
       This is a helpful tip
   
   !!! warning "Important"
       This is a warning
   ```

4. **Cross-reference** other pages:
   ```markdown
   See the [Installation Guide](getting-started/installation.md)
   ```

5. **API Documentation** using mkdocstrings:
   ```markdown
   ::: pytorch_segmentation_models_trainer.main
       options:
         show_source: true
   ```

### Adding New Pages

1. Create the markdown file in appropriate directory
2. Add to navigation in `mkdocs.yml`:
   ```yaml
   nav:
     - New Section:
       - New Page: path/to/new-page.md
   ```

## Auto-generated Content

Some content is automatically generated:

- **API Reference**: From docstrings using mkdocstrings
- **Navigation**: From `mkdocs.yml` structure
- **Search Index**: Automatically built by MkDocs

## Customization

### Theme Customization

Edit `mkdocs.yml`:

```yaml
theme:
  name: material
  palette:
    primary: blue
    accent: blue
  features:
    - navigation.tabs
    - search.highlight
```

### Adding Extensions

Add to `mkdocs.yml`:

```yaml
markdown_extensions:
  - pymdownx.superfences
  - admonition
  - tables
```

## Contributing

1. Write clear, concise documentation
2. Include working examples
3. Test all code snippets  
4. Check links work correctly
5. Preview locally before submitting

## Troubleshooting

### Common Issues

**Missing dependencies**:
```bash
pip install mkdocs-material mkdocstrings[python]
```

**API docs not generating**:
- Check docstrings exist in Python files
- Verify module paths in `::: module.name` references

**Build errors**:
- Check YAML syntax in `mkdocs.yml`
- Verify all referenced files exist
- Check markdown syntax

**Broken links**:
- Use relative paths: `[link](../other-page.md)`
- Check file extensions match exactly