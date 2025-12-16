import nextMDX from '@next/mdx'

basePath: '/gaussian-guy',
assetPrefix: '/gaussian-guy/',

const withMDX = nextMDX({
  extension: /\.mdx$/
})

export default withMDX({
  pageExtensions: ['js', 'jsx', 'mdx'],
  output: 'export',
  images: { unoptimized: true }
})
