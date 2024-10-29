module.exports = {
  singleQuote: true,
  semi: true,
  useTabs: false,
  tabWidth: 2,
  printWidth: 80,
  plugins: ["prettier-plugin-tailwindcss"],
  tailwindcss:{
    config:'./tailwind.config.js',
  }
};
