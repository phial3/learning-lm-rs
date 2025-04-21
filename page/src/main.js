import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import Toast from 'vue-toastification'
import 'vue-toastification/dist/index.css' // 引入样式

const app = createApp(App)

const options = {
    timeout: 2500, // 默认 toast 显示时间
    closeOnClick: true, // 点击 toast 关闭
    pauseOnFocusLoss: true, // 窗口失去焦点时暂停计时
    pauseOnHover: true, // 鼠标悬停时暂停计时
    draggable: true, // 可拖动
    draggablePercent: 0.6, // 拖动百分比
    showCloseButtonOnHover: false, // 鼠标悬停时显示关闭按钮
    hideProgressBar: false, // 隐藏进度条
    closeButton: 'button', // 关闭按钮类型
    icon: true, // 显示图标
    rtl: false // 从右到左布局
}

app.use(Toast, options)

app.mount('#app')
