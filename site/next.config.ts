import type { NextConfig } from 'next'
import createMDX from '@next/mdx'

const withMDX = createMDX({
  extension: /\.mdx?$/,
})

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/gaussian-guy',
  assetPrefix: '/gaussian-guy/',
  pageExtensions: ['ts', 'tsx', 'js', 'jsx', 'mdx'],
}

export default withMDX(nextConfig)