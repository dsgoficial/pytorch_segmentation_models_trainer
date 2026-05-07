// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'PyTorch Segmentation Models Trainer',
  tagline: 'Framework for training semantic segmentation models with PyTorch Lightning and Hydra',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://dsgoficial.github.io',
  baseUrl: '/pytorch_segmentation_models_trainer/',

  organizationName: 'dsgoficial',
  projectName: 'pytorch_segmentation_models_trainer',
  trailingSlash: true,

  onBrokenLinks: 'warn',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themes: [
    [
      // @ts-ignore -- string theme name is valid at runtime; @docusaurus/types tuple type is too strict here
      '@easyops-cn/docusaurus-search-local',
      /** @type {import('@easyops-cn/docusaurus-search-local').PluginOptions} */
      ({
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      }),
    ],
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/dsgoficial/pytorch_segmentation_models_trainer/tree/main/website/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'pytorch-smt',
        logo: {
          alt: 'PyTorch Segmentation Models Trainer',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            type: 'html',
            position: 'left',
            value: '<a class="navbar__link" href="/pytorch_segmentation_models_trainer/config-builder/">Config Builder</a>',
          },
          {
            href: 'https://github.com/dsgoficial/pytorch_segmentation_models_trainer',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {label: 'Getting Started', to: '/docs/getting-started/installation'},
              {label: 'Quick Start', to: '/docs/getting-started/quickstart'},
              {label: 'Configuration', to: '/docs/getting-started/configuration'},
            ],
          },
          {
            title: 'Tools',
            items: [
              {
                label: 'Config Builder',
                href: 'https://dsgoficial.github.io/pytorch_segmentation_models_trainer/config-builder/',
              },
              {
                label: 'PyPI',
                href: 'https://pypi.org/project/pytorch-segmentation-models-trainer/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/dsgoficial/pytorch_segmentation_models_trainer',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Philipe Borba. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'yaml', 'python'],
      },
    }),
};

export default config;
