module.exports = {
  content: ['./src/**/*.{html,ts,js,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#06B73B',
        background: '#F0F0F0',
        comment: '#00000061',
        danger: '#FF0000',
        'danger-light': '#FFE2E2',
        success: '#00BC4B',
        'success-light': '#E2FFEA',
      },
      fontFamily: {
        pretendard: ['Pretendard'],
      },
      fontSize: {
        xs: ['12px', '12px'],
        sm: ['14px', '14px'],
        base: ['16px', '16px'],
        lg: ['20px', '20px'],
        xl: ['24px', '24px'],
        '2xl': ['32px', '32px'],
      },
      borderRadius: {
        none: '0',
        DEFAULT: '10px',
        sm: '5px',
        md: '10px',
        lg: '20px',
        full: '100%',
      },
      width: {
        sign: '500px',
      },
      maxWidth: {
        sign: '500px',
      },
    },
  },
  plugins: [],
};
