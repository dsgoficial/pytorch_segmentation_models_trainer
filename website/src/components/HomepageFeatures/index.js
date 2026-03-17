import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Configuration-Driven',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Define your entire training pipeline with YAML files. Use Hydra for
        composable, reproducible experiments. Build configs visually with the{' '}
        <a href="/pytorch_segmentation_models_trainer/config-builder/">Config Builder</a>.
      </>
    ),
  },
  {
    title: 'Multiple Model Types',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Support for semantic segmentation, frame field models, object detection,
        instance segmentation, and Polygon-RNN — all through a unified interface.
      </>
    ),
  },
  {
    title: 'Advanced Polygonization',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Convert segmentation masks into precise vector polygons using active
        skeletons, active contours, simple polygonization, or Polygon-RNN.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
