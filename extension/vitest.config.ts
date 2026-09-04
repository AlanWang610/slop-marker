import { defineConfig } from "vitest/config";

export default defineConfig({
  // scripts/build.mjs substitutes this at bundle time (--model-base-url). Tests get the
  // empty string, so bundle-config.ts falls through to the published host -- which is the
  // value the shipped extension carries, and the one SHA256SUMS was generated against.
  define: { MODEL_HOST_OVERRIDE: '""' },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
