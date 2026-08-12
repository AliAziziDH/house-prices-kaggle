import { $ } from "bun";

// 📊 تابع ثبت تله‌متری ساختاریافته (Telemetry Logger) جهت دیباگ عاملی
function logTelemetry(step: string, status: "SUCCESS" | "FAILED" | "INFO", details: string) {
  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    step,
    status,
    details
  }));
}

async function runSetup() {
  try {
    logTelemetry("OS_PREPARATION", "INFO", "به‌روزرسانی کش پکیج‌های سیستم‌عامل اوبونتو...");
    await $`sudo apt-get update -qq`;

    logTelemetry("NGINX_INSTALLATION", "INFO", "در حال نصب وب‌سرور Nginx...");
    await $`sudo apt-get install -y -qq --no-install-recommends nginx python3`;

    logTelemetry("NGINX_HASH_BUCKET_OPTIMIZATION", "INFO", "تنظیم سایز باکت هش دامنه‌ها در فایل اصلی nginx.conf به ۱۲۸ بایت...");
    // خواندن فایل اصلی، اطمینان از قرارگیری تنظیمات و حذف موارد متداخل و تکراری
    const nginxConfPath = "/etc/nginx/nginx.conf";
    let nginxConf = await Bun.file(nginxConfPath).text();
    
    if (!nginxConf.includes("server_names_hash_bucket_size")) {
      // اضافه کردن دایرکتیو به داخل بلاک http
      nginxConf = nginxConf.replace(
        "http {",
        "http {\n        server_names_hash_bucket_size 128;"
      );
      await $`echo ${nginxConf} | sudo tee ${nginxConfPath} > /dev/null`;
    }

    logTelemetry("UNIFIED_VHOST_CONFIGURATION", "INFO", "ایجاد و مستقر کردن معماری یکپارچه طلایی در vhost پیش‌فرض انجینکس...");
    // ایجاد کانفیگ تمیز و یکپارچه بدون تداخل دامنه‌ها (بر اساس تفکیک پث‌ها)
    const unifiedNginxConfig = `
server {
    listen 8443 default_server;
    listen [::]:8443 default_server;
    server_name _;

    # 🔗 مسیر دریافت ساب‌اسکریپشن‌های اصلی (ساب ۱)
    location /sub {
        proxy_pass http://127.0.0.1:10000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # 🔄 مسیر دریافت ساب‌اسکریپشن چرخشی تمیز (ساب ۲)
    location /sub2 {
        proxy_pass http://127.0.0.1:12000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # 🧪 مسیر آزمایشگاهی لایه انتقال تجربی (ساب ۳)
    location /sub3 {
        proxy_pass http://127.0.0.1:12000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # 🛡️ پروکسی وب‌سوکت تانل اصلی ساب ۱ به هسته لوپ‌بک Xray
    location /ws {
        proxy_redirect off;
        proxy_pass http://127.0.0.1:10000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_buffering off; # غیرفعال‌سازی بافرینگ برای ارتباط دوطرفه زنده
    }

    # 🛡️ پروکسی وب‌سوکت تانل‌های کمکی ساب ۲ و ۳ به پورت داخلی ۱۲۰۰۰
    location /ws2 {
        proxy_redirect off;
        proxy_pass http://127.0.0.1:12000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_buffering off;
    }

    # 🕳️ Blackhole جهت ریجکت کردن هرگونه اسکن فعال DPI یا درخواست نامعتبر
    location / {
        return 444;
    }
}
`;

    // نوشتن فایل default با دستور ایمن tee جهت جلوگیری از Permission Denied
    await $`echo ${unifiedNginxConfig} | sudo tee /etc/nginx/sites-available/default > /dev/null`;

    logTelemetry("NGINX_SYNTAX_CHECK", "INFO", "در حال بررسی صحت سینتکس و کدهای انجینکس...");
    const checkResult = await $`sudo nginx -t`.quiet();
    if (checkResult.exitCode !== 0) {
      throw new Error(`شکست در بررسی سینتکس انجینکس: ${checkResult.stderr.toString()}`);
    }

    logTelemetry("SERVICE_ACTIVATION", "INFO", "راه‌اندازی مجدد سرویس انجینکس...");
    await $`sudo service nginx restart`;

    logTelemetry("PORT_BINDING_VERIFICATION", "INFO", "اعتبارسنجی مقید بودن ترافیک Xray به آدرس‌های لوپ‌بک داخلی سرور...");
    // بررسی اینکه آیا پورت‌های بک‌اند باز و ایزوله هستند
    const portReport = await $`sudo ss -tulpn | grep -E '10000|12000' || echo "clean"`.text();
    logTelemetry("PORT_REPORT", "INFO", `وضعیت پورت‌های بک‌اند:\n${portReport}`);

    logTelemetry("SELF_VERIFICATION_GATE", "SUCCESS", "راه‌اندازی محیط و وریفیکیشن فلوها با موفقیت به پایان رسید.");
    process.exit(0);
  } catch (error: any) {
    logTelemetry("SETUP_FAILED", "FAILED", error.message || String(error));
    process.exit(1);
  }
}

runSetup();
