export function useMDXComponents(components) {
  return {
    h1: (props) => <h1 style={{ marginTop: '2rem' }} {...props} />,
    p: (props) => <p style={{ margin: '1rem 0' }} {...props} />,
    ...components,
  }
}

