/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Configure rewrites so that frontend fetch('/api/predict') automatically
  // routes to the FastAPI backend during local development or when using proxying
  async rewrites() {
    const rawUrl = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").trim().replace(/\/+$/, "");
    // If backendUrl is relative (e.g. on same domain), no rewrite is needed
    if (rawUrl.startsWith("/")) {
      return [];
    }
    const cleanBase = rawUrl.endsWith("/api") ? rawUrl.slice(0, -4) : rawUrl;
    return [
      {
        source: "/api/:path*",
        destination: `${cleanBase}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
