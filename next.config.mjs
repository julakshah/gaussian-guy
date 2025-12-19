import withMDX from "@next/mdx";

const repoName = "gaussian-guy";
const isProd = process.env.NODE_ENV === "production";

const nextConfig = {
  reactStrictMode: true,
  pageExtensions: ["js", "jsx", "ts", "tsx", "mdx"],
  output: "export", // static export
  //basePath: isProd ? `/${repoName}` : "",
  //assetPrefix: isProd ? `/${repoName}/` : "",
  basePath: "/gaussian-guy",
  assetPrefix: "/gaussian-guy/",
  images: {
    unoptimized: true, // Required for static export
  },
};

// Wrap with MDX support
export default withMDX({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: [],
    rehypePlugins: [],
  },
})(nextConfig);
