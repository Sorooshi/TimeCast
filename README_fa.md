# بسته پیش‌بینی سری‌های زمانی

**🌍 زبان‌ها:** [English](README.md) | [Русский](README_ru.md) | [فارسی](README_fa.md)

بسته‌ای جامع بر اساس PyTorch برای پیش‌بینی سری‌های زمانی که چندین مدل پیشرفته یادگیری عمیق را با تنظیم خودکار فراپارامترها، مدیریت آزمایش‌ها و ردیابی قوی نتایج پیاده‌سازی می‌کند. **ریاضی تأیید شده** با فرمول‌بندی رسمی LaTeX و مطابقت کامل ابعاد.

## 🚀 ویژگی‌های کلیدی

- **چندین مدل پیشرفته**: LSTM، TCN، Transformer، HybridTCNLSTM، MLP
- **تنظیم خودکار فراپارامترها**: با استفاده از Optuna برای جستجوی بهینه پارامترها
- **مدیریت آزمایش‌ها**: ردیابی سازمان‌یافته آزمایش‌ها با توضیحات سفارشی
- **3 حالت آموزش**: گردکار ساده‌شده برای کاربردهای مختلف
- **پردازش قوی داده**: پیش‌پردازش تمیز و کارآمد بدون ویژگی‌های زمانی مصنوعی
- **تقسیم حرفه‌ای داده**: تقسیم train/val برای tune/train، فایل‌های تست جداگانه برای predict
- **تولید خودکار نمودار**: منحنی‌های آموزش/اعتبارسنجی به طور خودکار ذخیره می‌شوند
- **تاریخچه کامل آموزش**: ردیابی معیارها و پیشرفت بر اساس دوره
- **پیش‌پردازش داده‌های تاجران**: پایپ‌لاین کامل برای تبدیل تراکنش به سری زمانی
- **تأیید ریاضی**: سازگاری فرمول‌بندی LaTeX تأیید شده
- **گزارش‌گیری جامع**: ثبت فایل‌های دقیق برای اشکال‌زدایی و تحلیل
- **پشتیبانی چند پلتفرمی**: ایجاد مجموعه‌ای قوی در سیستم‌عامل‌های مختلف
- **تصویرسازی غنی**: منحنی‌های آموزش و نمودارهای ارزیابی
- **معماری ماژولار**: ساختار کد تمیز و قابل نگهداری

## 📐 مبانی ریاضی

این بسته فرمول‌بندی پیش‌بینی سری زمانی شرح‌داده‌شده در مقاله تحقیقاتی ما را پیاده‌سازی می‌کند:

### فرمول‌بندی مسئله
با توجه به داده‌های تراکنش سطح تاجر، ما مصرف کل را با استفاده از دنباله‌های تاریخی پیش‌بینی می‌کنیم:

**نقشه‌برداری LaTeX → پیاده‌سازی:**
- دنباله تاریخی: $\mathcal{H}_t \in \mathbb{R}^{(k+1) \times N}$ ↔ `(sequence_length, n_features)`
- مصرف تاجر: $X_t \in \mathbb{R}^N$ ↔ `merchant_features[t]`
- پیش‌بینی هدف: $y_t = \sum_{m=1}^N x_{m,t}$ ↔ `np.sum(data[t])`

**✅ سازگاری ابعاد تأیید شده:**
```
LaTeX: 𝒽_t ∈ ℝ^{(k+1)×N}  ↔  پیاده‌سازی: (batch_size, sequence_length, n_features)
```

## 📊 مدل‌های پیاده‌سازی‌شده

| مدل | شرح | مورد استفاده | مرجع مقاله |
|-----|-----|--------------|-------------|
| **LSTM** | شبکه حافظه کوتاه-مدت طولانی | یادگیری الگوی ترتیبی | Hochreiter & Schmidhuber (1997) |
| **TCN** | شبکه کانولوشنال زمانی | استخراج ویژگی سلسله‌مراتبی | Bai et al. (2018) |
| **Transformer** | مدل مبتنی بر خودتوجهی | وابستگی‌های زمانی پیچیده | Vaswani et al. (2017) |
| **HybridTCNLSTM** | ترکیب TCN + LSTM | بهترین از هر دو معماری | پیاده‌سازی سفارشی |
| **MLP** | پرسپترون چندلایه | مقایسه خط پایه | Zhang et al. (1998) |

## 🛠️ نصب

1. **کلون کردن مخزن:**
```bash
git clone https://github.com/Sorooshi/TimeCast.git
cd TimeCast
```

2. **ایجاد و فعال‌سازی محیط مجازی:**
```bash
python -m venv venv
source venv/bin/activate  # در ویندوز: venv\Scripts\activate
```

3. **نصب وابستگی‌ها:**
```bash
pip install -r requirements.txt
```

## 📖 استفاده

### شروع سریع با داده‌های تاجران

برای پیش‌پردازش داده‌های تراکنش تاجران (نقطه شروع توصیه‌شده):

```bash
# مرحله 1: اجرای مثال پیش‌پردازش
python example.py

# مرحله 2: آموزش مدل‌ها روی داده‌های پیش‌پردازش‌شده با همه آرگومان‌ها
python main.py --model Transformer \
               --data_name merchant_processed \
               --data_path data/merchant_processed.csv \
               --mode train \
               --train_tuned false \
               --experiment_description "merchant_baseline" \
               --n_trials 100 \
               --epochs 100 \
               --patience 25 \
               --sequence_length 5
```

### رابط خط فرمان

بسته یک CLI جامع با 3 حالت متمایز ارائه می‌دهد:

```bash
python main.py --model <MODEL_NAME> \
               --data_name <DATASET_NAME> \
               --mode <MODE> \
               --experiment_description <DESCRIPTION> \
               [گزینه‌های اضافی]
```

### 🎯 حالت‌های آموزش

| حالت | شرح | استفاده از داده | مصنوعات ذخیره‌شده |
|------|-----|---------------|------------------|
| `tune` | فقط بهینه‌سازی فراپارامترها | تقسیم train/val از داده‌های اصلی | پارامترهای تنظیم‌شده در ساختار سلسله‌مراتبی |
| `train` | آموزش با پارامترهای تنظیم‌شده (`--train_tuned true`) یا پیش‌فرض (`--train_tuned false`) | تقسیم train/val از داده‌های اصلی | **نمودارها، تاریخچه، معیارها، پیش‌بینی‌ها** در دایرکتوری‌های سلسله‌مراتبی |
| `predict` | بارگیری مدل آموزش‌دیده و پیش‌بینی (`--predict_tuned true/false`) | **نیاز به فایل داده تست جداگانه** | پیش‌بینی‌ها و معیارها در ساختار سلسله‌مراتبی |
| `report` | نمایش تحلیل جامع آزمایش‌ها | - | خلاصه‌های تحلیل |

**🎨 سازماندهی فایل‌ها**: تمام مصنوعات اکنون در ساختار سلسله‌مراتبی ذخیره می‌شوند:
```
Results/{model}/{mode}/{exp_subdir}/
History/{model}/{mode}/{exp_subdir}/
Predictions/{model}/{mode}/{exp_subdir}/
Metrics/{model}/{mode}/{exp_subdir}/
Hyperparameters/{model}/{mode}/{exp_subdir}/
Plots/{model}/{mode}/{exp_subdir}/
Logs/{model}/{mode}/{exp_subdir}/
Weights/{model}/{mode}/{exp_subdir}/
```
که در آن `{exp_subdir}` معمولاً به صورت `seq_len_{N}/` یا `seq_len_{N}/{experiment_description}/` (و برای حالت پیش‌بینی ممکن است `test_{test_data_name}` نیز اضافه شود) ساخته می‌شود.

**🔄 ایجاد دایرکتوری بر اساس حالت**:
- `train_tuned/train_default`: ایجاد results، history، plots، predictions، metrics
- `predict`: فقط ایجاد results، predictions، metrics
- `tune`: ایجاد تمام دایرکتوری‌ها شامل hyperparameters

### 📊 قابلیت‌های حالت گزارش

حالت گزارش تحلیل جامعی از آزمایش‌های شما با چندین گزینه نمایش ارائه می‌دهد:

```bash
python main.py --model <MODEL> --data_name <DATA> --mode report --report_type <TYPE>
```

#### انواع گزارش‌های موجود:

| نوع گزارش | شرح | نمایش |
|-----------|-----|--------|
| `all` | گزارش جامع کامل | همه چیز با هم |
| `models` | مدل‌های آموزش‌دیده موجود | وضعیت مدل‌ها، کمال |
| `performance` | جداول مقایسه عملکرد | بهترین معیارها، رتبه‌بندی |
| `best` | بهترین پیکربندی‌ها | 5 پیکربندی برتر با فراپارامترها |
| `timeline` | خط زمانی آزمایش‌ها | تاریخچه زمانی آزمایش‌ها |
| `files` | مسیرهای فایل و اطلاعات ذخیره‌سازی | ساختار دایرکتوری، اندازه فایل‌ها |

#### 🔍 نمونه‌های گزارش:

**نمایش همه مدل‌های موجود:**
```bash
python main.py --model LSTM --data_name test_data --mode report --report_type models
```

**مقایسه عملکرد:**
```bash
python main.py --model LSTM --data_name test_data --mode report --report_type performance
```

**تحلیل کامل:**
```bash
python main.py --model LSTM --data_name test_data --mode report --report_type all
```

#### 📈 آنچه هر گزارش نشان می‌دهد:

**🤖 گزارش مدل‌ها:**
- مدل‌های آموزش‌دیده موجود با وضعیت (کامل/جزئی/بدون وزن)
- در دسترس بودن مدل‌های تنظیم‌شده و پیش‌فرض
- آمار کمال مدل‌ها
- نمای کلی سازماندهی آزمایش‌ها

**📊 گزارش عملکرد:**
- بهترین عملکرد بر اساس نوع مدل
- رتبه‌بندی‌های دقیق عملکرد
- مقایسه اتلاف تست، R² و MAPE
- روندهای عملکرد در آزمایش‌ها

**🏆 گزارش بهترین پیکربندی‌ها:**
- 5 پیکربندی برتر
- فراپارامترهای کلیدی برای بهترین مدل‌ها
- معیارهای عملکرد برای هر پیکربندی
- توصیه‌های فراپارامتر

**⏰ گزارش خط زمانی:**
- تاریخچه زمانی آزمایش‌ها
- برچسب‌های زمانی تغییر فایل
- تحلیل فراوانی آزمایش
- خلاصه محدوده تاریخ

**📁 گزارش فایل‌ها:**
- ساختار کامل دایرکتوری
- فایل‌های فراپارامتر با اندازه‌ها و تاریخ‌ها
- فایل‌های وزن با اطلاعات ذخیره‌سازی
- سازماندهی فایل‌های نتایج
- آمار کل استفاده از ذخیره‌سازی

### 📋 آرگومان‌ها

#### آرگومان‌های ضروری
- `--model`: نام مدل (LSTM, TCN, Transformer, HybridTCNLSTM, MLP)
- `--data_name`: نام مجموعه داده (بدون پسوند .csv)

#### آرگومان‌های اختیاری
- `--data_path`: مسیر کامل فایل داده (پیش‌فرض: data/{data_name}.csv)
- `--mode`: حالت آموزش (پیش‌فرض: train)
- `--experiment_description`: توضیح آزمایش سفارشی (پیش‌فرض: seq_len_{sequence_length})
- `--train_tuned`: استفاده از پارامترهای تنظیم‌شده برای آموزش (true/false، پیش‌فرض: true)
- `--predict_tuned`: استفاده از مدل تنظیم‌شده برای پیش‌بینی (true/false، پیش‌فرض: true)
- `--report_type`: نوع گزارش برای نمایش (all/models/performance/best/timeline/files، پیش‌فرض: all)
- `--n_trials`: آزمایش‌های تنظیم فراپارامتر (پیش‌فرض: 100)
- `--epochs`: دوره‌های آموزش (پیش‌فرض: 100)
- `--patience`: صبر توقف زودهنگام (پیش‌فرض: 25)
- `--sequence_length`: طول دنباله ورودی (پیش‌فرض: 10)
- `--k_folds`: تعداد تاهای K-fold اعتبارسنجی متقابل (پیش‌فرض: 5)

### 🔧 مهم: استفاده از مسیر داده

**اشتباه رایج:** `--data_path` را به یک دایرکتوری اشاره نکنید!

```bash
# ❌ اشتباه - این کار نخواهد کرد
python main.py --model LSTM --data_name my_data --data_path data/

# ✅ درست - مسیر کامل فایل را مشخص کنید
python main.py --model LSTM --data_name my_data --data_path data/my_data.csv

# ✅ توصیه می‌شود - اجازه دهید سیستم مسیر را خودکار بسازد
python main.py --model LSTM --data_name my_data
# این به طور خودکار استفاده می‌کند: data/my_data.csv
```

**نکات کلیدی:**
- `--data_path` انتظار **مسیر فایل** دارد، نه دایرکتوری
- اگر مشخص نشود، سیستم می‌سازد: `data/{data_name}.csv`
- همیشه پسوند `.csv` را هنگام مشخص کردن `--data_path` اضافه کنید

## 🏪 پیش‌پردازش داده‌های تاجران

### پایپ‌لاین پیش‌پردازش (`example.py`)

پایپ‌لاین کامل برای تبدیل داده‌های تراکنش خام تاجران به فرمت سری زمانی:

```bash
python example.py
```

**مراحل پایپ‌لاین:**
1. **بارگیری داده‌های تراکنش**: بارگیری داده‌های سطح تراکنش خام
2. **تجمیع تاجران**: گروه‌بندی بر اساس دوره‌های زمانی و تاجران
3. **ویژگی‌های زمینه‌ای**: اضافه کردن ویژگی‌های مبتنی بر زمان (فصلی، تعطیلات و غیره)
4. **سازگاری LaTeX**: اطمینان از مطابقت ابعاد
5. **تأیید**: تست با TimeSeriesPreprocessor

**فرمت ورودی:**
```csv
timestamp,merchant_id,customer_id,amount,day_of_week,hour,is_weekend,is_holiday,transaction_speed,customer_loyalty_score
2023-01-01 03:41:00,1,23,16.02,6,3,True,False,8.87,79.8
2023-01-01 06:28:00,4,25,99.56,6,6,True,False,5.9,48.8
...
```

**فرمت خروجی:**
```csv
date,merchant_1,merchant_2,merchant_3,merchant_4,merchant_5,hour,day_of_week,is_weekend,month,day_of_month,sin_month,cos_month,sin_hour,cos_hour,is_holiday
2023-01-01,454.17,207.98,216.56,460.11,644.78,0,5,1.0,1,1,0.0,1.0,0.0,1.0,1.0
2023-01-02,423.89,189.45,234.12,501.23,678.91,0,0,0.0,1,2,0.0,1.0,0.0,1.0,0.0
...
```

## 🧪 تست و تأیید

### مجموعه تست جامع

فراخوانی کامل تست‌های اعتبارسنجی:

```bash
cd Test/
python test_script.py
```

**نتایج تست:** ✅ 100% موفقیت (11/11 تست)

### تست‌های تأیید ریاضی

- ✅ **تأیید ابعاد**: LaTeX ↔ پیاده‌سازی
- ✅ **محاسبه هدف**: `np.sum()` درست
- ✅ **فرمت داده**: تطابق بعد کامل
- ✅ **یکپارچگی پایپ‌لاین**: پردازش کامل

## 💡 گردکارهای مثال

### 1. گردکار کامل داده‌های تاجران

```bash
# مرحله 1: پیش‌پردازش داده‌های تاجران
python example.py

# مرحله 2: تنظیم فراپارامتر
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode tune \
               --experiment_description "merchant_baseline" \
               --n_trials 50 \
               --epochs 100 \
               --sequence_length 5

# مرحله 3: آموزش با پارامترهای تنظیم‌شده (K-fold CV)
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode train \
               --train_tuned true \
               --experiment_description "merchant_tuned" \
               --epochs 100 \
               --sequence_length 5

# مرحله 4: مقایسه با پارامترهای پیش‌فرض
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode apply_not_tuned \
               --experiment_description "merchant_default" \
               --epochs 100 \
               --sequence_length 5

# مرحله 5: مشاهده همه نتایج
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode report \
               --experiment_description "merchant_baseline"
```

### 2. گردکار تست سریع

```bash
# تست سریع با پارامترهای پیش‌فرض
python main.py --model LSTM \
               --data_name my_data \
               --mode apply_not_tuned \
               --experiment_description "quick_test" \
               --epochs 20 \
               --sequence_length 5
```

## 📁 ساختار پروژه

```
TimeCast/
├── data/                    # فایل‌های مجموعه داده
├── models/                  # پیاده‌سازی مدل‌ها
├── utils/                   # ابزارهای کمکی
├── History/                 # تاریخچه آموزش
├── Hyperparameters/         # پارامترهای تنظیم‌شده
├── Results/                 # خلاصه نتایج
├── Plots/                   # تصویرسازی‌ها
├── Logs/                    # فایل‌های ثبت
├── Test/                    # مجموعه تست‌ها
├── main.py                  # نقطه ورود اصلی
├── example.py               # پیش‌پردازش تاجران
└── requirements.txt         # وابستگی‌ها
```

## 🤝 مشارکت

1. فورک پروژه
2. شاخه ویژگی ایجاد کنید (`git checkout -b feature/AmazingFeature`)
3. تغییرات خود را کامیت کنید (`git commit -m 'Add some AmazingFeature'`)
4. به شاخه پوش کنید (`git push origin feature/AmazingFeature`)
5. Pull Request باز کنید

## 📝 مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات بیشتر فایل [LICENSE](LICENSE) را ببینید.

## 📧 تماسs

سروش شلیله - [sorooshshalileh@example.com](mailto:sr.shalileh@gmail.com)

لینک پروژه: [https://github.com/Sorooshi/TimeCast](https://github.com/Sorooshi/TimeCast)

## 🎖️ تشکر

- تیم [PyTorch](https://pytorch.org/) برای فریمورک عالی
- توسعه‌دهندگان [Optuna](https://optuna.org/) برای کتابخانه تنظیم فراپارامتر
- جامعه متن‌باز برای ابزارها و کتابخانه‌های بی‌نظیر

---

**📊 آماده برای تحقیق و تولید** | **🔬 تأیید ریاضی** | **🌍 پشتیبانی چندزبانه** 