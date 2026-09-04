// @ts-check
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "src/assets/**"] },
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    rules: {
      // The ports mirror Python control flow line for line; readability there beats
      // idiomatic-JS rewrites, and the fixtures are the real guard.
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/consistent-type-imports": "error",
    },
  },
  {
    files: ["**/*.js", "**/*.mjs"],
    extends: [tseslint.configs.disableTypeChecked],
    languageOptions: {
      globals: {
        console: "readonly",
        process: "readonly",
        performance: "readonly",
        fetch: "readonly",
        URL: "readonly",
        // e2e/run.mjs passes callbacks to page.evaluate(); their bodies are serialised and
        // run in the browser, so they legitimately reference DOM and extension globals.
        document: "readonly",
        window: "readonly",
        self: "readonly",
        CSS: "readonly",
        chrome: "readonly",
        setTimeout: "readonly",
      },
    },
  },
);
