import nextMDX from '@next/mdx'

const withMDX = nextMDX({
  extension: /\.mdx$/
})

export default withMDX({
  pageExtensions: ['js', 'jsx', 'mdx'],
  output: 'export',
  images: { unoptimized: true },

  // GitHub Pages project path
  basePath: '/gaussian-guy',
  assetPrefix: '/gaussian-guy/',
})

