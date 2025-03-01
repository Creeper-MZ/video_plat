module.exports = {
    extends: [
        'react-app',
        'react-app/jest'
    ],
    plugins: [
        'jest'
    ],
    env: {
        browser: true,
        node: true,
        es6: true,
        'jest/globals': true
    },
    rules: {
        // 如果需要自定义规则，可以在这里添加
    }
};