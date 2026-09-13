/** @type {import('next').NextConfig} */
const nextConfig = {
  // Avoid Next.js writing generated AGENTS.md / CLAUDE.md files into this
  // vendored app's directory on every `next dev` / `next build`.
  agentRules: false,
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
