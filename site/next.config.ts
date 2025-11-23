import type { NextConfig } from 'next'
import createMDX from '@next/mdx'

const isProd = process.env.NODE_ENV === 'production'

const withMDX = createMDX({
  extension: /\.mdx?$/,
})

const nextConfig: NextConfig = {
  output: 'export',
  basePath: isProd ? '/gaussian-guy' : '',
  assetPrefix: isProd ? '/gaussian-guy/' : '',
  pageExtensions: ['ts', 'tsx', 'js', 'jsx', 'mdx'],
}

export default withMDX(nextConfig)