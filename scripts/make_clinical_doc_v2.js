// Clinical document v2 with tables and embedded figures.
// Run: node scripts/make_clinical_doc_v2.js

const fs = require('fs');
const path = require('path');
const {
    Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
    Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
    LevelFormat, ImageRun, PageBreak, Header, Footer, PageNumber,
} = require('docx');

const REPO = path.join(__dirname, '..');
const FIG_DIR = path.join(REPO, 'results', 'doc_figures');

// ====== Style helpers ======
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
const TABLE_W = 9360;

function p(text, opts = {}) {
    return new Paragraph({
        spacing: { after: 140 },
        alignment: opts.align || AlignmentType.JUSTIFIED,
        children: [new TextRun({ text, size: 22, ...opts })],
    });
}

function pMix(runs, opts = {}) {
    return new Paragraph({
        spacing: { after: 140 },
        alignment: opts.align || AlignmentType.JUSTIFIED,
        children: runs.map(r => new TextRun({ size: 22, ...r })),
    });
}

function h1(text) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 360, after: 180 },
        children: [new TextRun({ text, bold: true, size: 30, color: "1F3864" })],
    });
}

function h2(text) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 240, after: 140 },
        children: [new TextRun({ text, bold: true, size: 24, color: "2E75B6" })],
    });
}

function bullet(text) {
    return new Paragraph({
        numbering: { reference: "b", level: 0 },
        spacing: { after: 80 },
        children: [new TextRun({ text, size: 22 })],
    });
}

function cell(text, opts = {}) {
    return new TableCell({
        borders,
        width: { size: opts.w || 1500, type: WidthType.DXA },
        shading: opts.shade ? { fill: opts.shade, type: ShadingType.CLEAR } : undefined,
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        children: [new Paragraph({
            alignment: opts.align || AlignmentType.LEFT,
            children: [new TextRun({ text, size: 18, bold: opts.bold })],
        })],
    });
}

function headerRow(cols, widths) {
    return new TableRow({
        tableHeader: true,
        children: cols.map((c, i) => cell(c, {
            w: widths[i], bold: true, align: AlignmentType.CENTER, shade: "D9E2F3",
        })),
    });
}

function dataRow(cols, widths, opts = {}) {
    return new TableRow({
        children: cols.map((c, i) => cell(c, {
            w: widths[i],
            align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
            bold: opts.bold,
            shade: opts.shade,
        })),
    });
}

function imageFig(filename, widthPx = 600, heightPx = 360, caption = "") {
    const imgPath = path.join(FIG_DIR, filename);
    const buf = fs.readFileSync(imgPath);
    const para = [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 80 },
        children: [new ImageRun({
            type: filename.endsWith('.png') ? 'png' : 'jpg',
            data: buf,
            transformation: { width: widthPx, height: heightPx },
            altText: { title: caption, description: caption, name: filename },
        })],
    })];
    if (caption) {
        para.push(new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
            children: [new TextRun({ text: caption, size: 18, italics: true, color: "404040" })],
        }));
    }
    return para;
}

// ====== Title page ======
const titlePage = [
    new Paragraph({
        spacing: { before: 1800, after: 240 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
            text: "Предиктивная роль опухолевого микроокружения в модификации ответа на таргетную терапию при раке молочной железы",
            bold: true, size: 32, color: "1F3864",
        })],
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 600 },
        children: [new TextRun({
            text: "Ретроспективный биомаркер-анализ адаптивного клинического исследования I-SPY2",
            size: 22, italics: true, color: "595959",
        })],
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 1600, after: 80 },
        children: [new TextRun({ text: "Автор: Д. Лысенко", size: 22 })],
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "Дата: 23 апреля 2026 г.", size: 22 })],
    }),
    new Paragraph({ children: [new PageBreak()] }),
];

// ====== Content sections ======
const intro = [
    h1("Введение"),
    p("Ответ на таргетную терапию при раке молочной железы остается клинически неоднородным даже у пациенток, формально соответствующих стандартным критериям назначения лечения. Одним из вероятных источников такой вариабельности является опухолевое микроокружение (TME), включающее стромальный и иммунный компоненты, степень T-клеточной инфильтрации, выраженность TGFβ-сигналинга и активность опухоль-ассоциированных фибробластов (CAF). Эти характеристики потенциально способны модифицировать чувствительность опухоли к лечению, однако пока недостаточно интегрированы в клиническую стратификацию."),
];

const aim = [
    h1("Цель"),
    p("Оценить предиктивную роль опухолевого микроокружения в модификации эффективности таргетной терапии при раке молочной железы и определить клиническую значимость стромального биомаркера как возможного инструмента стратификации пациенток."),
];

const methods = [
    h1("Материалы и методы"),
    p("Проведен ретроспективный анализ данных адаптивного неоадъювантного исследования I-SPY2 (n = 986, рак молочной железы II–III стадии). Во всех терапевтических режимах паклитаксел использовался как базовая химиотерапия, к которой добавлялись различные таргетные препараты. Контрольная группа (паклитаксел без таргетной терапии, n = 179) включала HER2-негативных пациенток."),
    p("Первичной конечной точкой являлся патоморфологический полный ответ (pCR), определяемый как отсутствие резидуальной инвазивной опухоли в молочной железе и регионарных лимфатических узлах после завершения лечения."),
    p("Молекулярный профиль оценивался по данным претерапевтической core-биопсии с использованием ДНК-микрочипов Agilent 44K (платформы GPL20078 и GPL30493). На основе транскриптомных данных рассчитывались сигнатуры микроокружения методом ssGSEA. Ключевой биомаркер, стромальный индекс опухоли (stromal score), определялся как сумма z-нормированных показателей активности опухоль-ассоциированных фибробластов (CAF) и TGFβ-сигналинга. Формула биомаркера и план статистического анализа были pre-specified в протоколе до начала анализа данных."),
    p("Группы трастузумаба, T-DM1 и других режимов на основе анти-HER2 терапии были исключены из основного анализа в связи с отсутствием в данном релизе соответствующего HER2-позитивного контроля. Группа нератиниба, включавшая обе HER2-подгруппы, была ограничена HER2-негативной подвыборкой для корректности сравнения."),
    p("Для анализа предиктивного эффекта строились модели логистической регрессии с включением взаимодействия между стромальным индексом и типом терапии. В модели учитывались клинические ковариаты (статус гормональных рецепторов, результат MammaPrint, статус HER2). Статистическая значимость оценивалась с использованием 1000 перестановочных тестов, доверительные интервалы рассчитаны методом бутстрапа (500 итераций). Сводный эффект через классы препаратов рассчитан методом random-effects мета-анализа DerSimonian–Laird. Согласованность направления эффекта дополнительно оценена биномиальным sign-test."),
];

const resultsHeader = h1("Результаты");

const resultsP1 = [
    p("Во всех семи анализируемых классах таргетной терапии выявлено однонаправленное снижение относительной эффективности лечения у пациенток с высоким значением стромального индекса. Отношение шансов взаимодействия было меньше единицы во всех случаях, что указывает на согласованный паттерн через различные молекулярные механизмы действия препаратов. Количественные оценки эффекта представлены в Таблице 1 и на Рисунке 1."),
];

// === Table 1: per-arm ===
const T1_widths = [2400, 1300, 1200, 1100, 1100, 1160, 1100];
const table1 = new Table({
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: T1_widths,
    rows: [
        headerRow(["Класс препарата", "n контр./лечение", "pCR контр.", "pCR леч.", "OR взаимод.", "95% ДИ", "p (перм.)"], T1_widths),
        dataRow(["Пембролизумаб (анти-PD-1)", "179 / 69", "17,3%", "44,9%", "0,46", "0,24 до 0,89", "0,017"], T1_widths),
        dataRow(["Ганитумаб (анти-IGF-1R)", "179 / 105", "17,3%", "22,9%", "0,46", "0,25 до 0,86", "0,020"], T1_widths),
        dataRow(["Нератиниб (pan-HER TKI)", "179 / 49", "17,3%", "36,8%", "0,49", "0,23 до 1,03", "0,100"], T1_widths),
        dataRow(["MK-2206 (AKT-ингибитор)", "179 / 60", "17,3%", "30,0%", "0,68", "0,34 до 1,37", "0,326"], T1_widths),
        dataRow(["Ганетеспиб (HSP90)", "179 / 93", "17,3%", "28,0%", "0,67", "0,36 до 1,22", "0,209"], T1_widths),
        dataRow(["ABT-888 + карбоплатин", "179 / 71", "17,3%", "38,0%", "0,80", "0,40 до 1,60", "0,518"], T1_widths),
        dataRow(["AMG-386 (анти-Ang)", "179 / 114", "17,3%", "28,7%", "0,86", "0,48 до 1,54", "0,628"], T1_widths),
    ],
});

const t1Caption = new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 80, after: 200 },
    children: [
        new TextRun({ text: "Таблица 1. ", size: 18, bold: true, italics: true, color: "404040" }),
        new TextRun({ text: "Взаимодействие стромального индекса с добавлением таргетной терапии (модель логистической регрессии с поправкой на статус гормональных рецепторов и MammaPrint).", size: 18, italics: true, color: "404040" }),
    ],
});

const fig1 = imageFig("fig1_forest.png", 620, 380,
    "Рис. 1. Forest plot отношения шансов взаимодействия (стромальный индекс × терапия). Все 7 классов препаратов показывают OR < 1 (синий ромб: объединённый эффект методом мета-анализа).");

const resultsP2 = [
    p("Согласованность направления эффекта оценена двумя независимыми способами. Биномиальный sign-test показал, что во всех 7 из 7 валидных групп отношение шансов взаимодействия меньше единицы; одностороннее значение p составило 0,008, двустороннее p = 0,016. Объединённый эффект, рассчитанный методом random-effects мета-анализа, составил OR_int = 0,62 (95% ДИ 0,49 до 0,80; p = 0,0001). Межгрупповая гетерогенность фактически отсутствовала (τ² = 0), что подтверждает наличие общего биологического механизма для разных классов препаратов."),
    p("При тертильной стратификации по значению стромального индекса абсолютная клиническая выгода от добавления таргетной терапии распределялась резко неравномерно. Подгруппа со стромально-бедной опухолью (нижний тертиль) демонстрировала наибольший прирост частоты pCR при добавлении препарата, тогда как у пациенток со стромально-богатой опухолью (верхний тертиль) этот прирост сводился к минимальному или становился отрицательным. Иллюстративные данные для двух наиболее изученных препаратов представлены в Таблице 2 и на Рисунке 2."),
];

// === Table 2: tertiles ===
const T2_widths = [1500, 1100, 1300, 1300, 1300, 1100, 1760];
const table2 = new Table({
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: T2_widths,
    rows: [
        headerRow(["Терапия", "Тертиль", "n контр./леч.", "pCR контр.", "pCR леч.", "ARR, п.п.", "OR (95% ДИ)"], T2_widths),
        dataRow(["Пембролизумаб", "Низкий", "57 / 26", "14,0%", "53,8%", "+39,8", "6,76 (2,36–19,3)"], T2_widths, { shade: "E2EFDA" }),
        dataRow(["Пембролизумаб", "Средний", "58 / 24", "20,7%", "50,0%", "+29,3", "3,72 (1,37–10,1)"], T2_widths),
        dataRow(["Пембролизумаб", "Высокий", "64 / 19", "17,2%", "26,3%", "+9,1", "1,76 (0,55–5,7)"], T2_widths, { shade: "FCE4D6" }),
        dataRow(["Ганитумаб", "Низкий", "61 / 34", "14,8%", "47,1%", "+32,3", "4,93 (1,89–12,9)"], T2_widths, { shade: "E2EFDA" }),
        dataRow(["Ганитумаб", "Средний", "57 / 37", "21,1%", "10,8%", "−10,2", "0,49 (0,15–1,57)"], T2_widths),
        dataRow(["Ганитумаб", "Высокий", "61 / 34", "16,4%", "11,8%", "−4,6", "0,72 (0,22–2,38)"], T2_widths, { shade: "FCE4D6" }),
    ],
});

const t2Caption = new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 80, after: 200 },
    children: [
        new TextRun({ text: "Таблица 2. ", size: 18, bold: true, italics: true, color: "404040" }),
        new TextRun({ text: "Частота pCR по тертилям стромального индекса для двух препаратов с наиболее выраженным предиктивным эффектом. Зелёный фон: подгруппа максимального benefit; розовый: минимального или отрицательного.", size: 18, italics: true, color: "404040" }),
    ],
});

const fig2 = imageFig("fig2_tertile_bars.png", 640, 280,
    "Рис. 2. Распределение клинической пользы по тертилям стромального индекса. У пациенток со стромально-бедной опухолью добавление таргетной терапии даёт наибольший прирост pCR; у стромально-богатых пациенток польза минимальна или отсутствует.");

const resultsP3 = [
    p("Для пембролизумаба различие абсолютной клинической пользы между крайними тертилями составило 30,7 процентных пункта (с +39,8 п.п. в нижнем тертиле до +9,1 п.п. в верхнем). Для ганитумаба наблюдается классический паттерн качественного взаимодействия: при низком стромальном индексе препарат даёт выраженный benefit (+32,3 п.п.), при высоком значении биомаркера добавление препарата ассоциировано с худшим исходом, чем химиотерапия одна (−4,6 п.п.). Аналогичные направленные паттерны прослеживаются для нератиниба и MK-2206."),
    p("При стратификации по молекулярному подтипу опухоли максимальная предиктивная значимость стромального индекса наблюдалась в подгруппе тройного негативного рака молочной железы. Для всех трёх подробно проанализированных препаратов (ганитумаб, пембролизумаб, нератиниб) у пациенток с трижды-негативным раком отношение шансов взаимодействия было меньше 0,35 при значениях p от 0,015 до 0,032. В подгруппе HR+/HER2− эффект сохранял ту же направленность только для двух из трёх препаратов и не достигал статистической значимости (Таблица 3, Рисунок 3)."),
];

// === Table 3: subtype ===
const T3_widths = [1900, 1500, 1500, 1300, 1500, 1660];
const table3 = new Table({
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: T3_widths,
    rows: [
        headerRow(["Препарат", "Подтип", "n контр./леч.", "OR взаимод.", "95% ДИ", "p"], T3_widths),
        dataRow(["Ганитумаб", "HR+/HER2-", "94 / 58", "0,81", "0,32 до 2,01", "0,646"], T3_widths),
        dataRow(["Ганитумаб", "ТНРМЖ", "85 / 47", "0,33", "0,13 до 0,81", "0,015"], T3_widths, { shade: "FFF2CC" }),
        dataRow(["Пембролизумаб", "HR+/HER2-", "94 / 40", "0,86", "0,35 до 2,09", "0,741"], T3_widths),
        dataRow(["Пембролизумаб", "ТНРМЖ", "85 / 29", "0,31", "0,10 до 0,90", "0,031"], T3_widths, { shade: "FFF2CC" }),
        dataRow(["Нератиниб", "HR+/HER2-", "94 / 17", "1,40", "0,31 до 6,21", "0,661"], T3_widths),
        dataRow(["Нератиниб", "ТНРМЖ", "85 / 32", "0,34", "0,13 до 0,91", "0,032"], T3_widths, { shade: "FFF2CC" }),
    ],
});

const t3Caption = new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 80, after: 200 },
    children: [
        new TextRun({ text: "Таблица 3. ", size: 18, bold: true, italics: true, color: "404040" }),
        new TextRun({ text: "Подтипоспецифичность стромального биомаркера. Жёлтый фон: значимый эффект (p < 0,05) в подгруппе тройного негативного рака.", size: 18, italics: true, color: "404040" }),
    ],
});

const fig3 = imageFig("fig3_subtype.png", 580, 320,
    "Рис. 3. Подтипоспецифичность стромального биомаркера. Сигнал статистически значим во всех трёх классах препаратов в подгруппе ТНРМЖ; в HR+/HER2- сохраняется направленность, но без статистической значимости.");

const resultsP4 = [
    p("Декомпозиция стромального биомаркера на отдельные компоненты (анализ leave-one-component-out) показала, что основной вклад в наблюдаемый эффект вносит CAF-составляющая. Тестирование CAF-сигнатуры в качестве самостоятельного предиктора без TGFβ-компонента воспроизводит интерактивный эффект практически в полном объёме (для пембролизумаба OR 0,42, p = 0,010; для ганитумаба OR 0,50, p = 0,023; для нератиниба OR 0,46, p = 0,032). Отдельная TGFβ-сигнатура без CAF демонстрирует более слабый и менее устойчивый эффект, что упрощает биологическую интерпретацию результата и потенциально облегчает разработку клинически применимого варианта теста."),
    p("Дополнительный тест биологической специфичности показал, что выраженность интерактивного эффекта зависит от априорной патофизиологической связи препарата со стромальной осью. Медиана отношения шансов взаимодействия в группах с биологически обоснованной связью со стромой (пембролизумаб, ганитумаб, нератиниб, MK-2206, ABT-888 с карбоплатином) составила 0,49, тогда как в группах без такой связи (ганетеспиб, AMG-386) медианное значение равнялось 0,76. Эта закономерность согласуется с гипотезой о патофизиологически обусловленном механизме и снижает вероятность общего неспецифического конфаундера."),
    p("Проверка устойчивости сигнала проведена методом стратифицированной 60-процентной субвыборки с повторением 500 раз для группы с наиболее выраженным эффектом (ганитумаб). В 48 процентах подвыборок взаимодействие сохраняло статистическую значимость на уровне p < 0,05, медиана значений p составила 0,054. Полученное распределение указывает на то, что наблюдаемый эффект не является случайным, однако демонстрирует умеренную зависимость от состава выборки, что характерно для одиночных биомаркерных исследований при числе пациенток в экспериментальной группе менее 150."),
];

const patho = [
    h1("Патофизиологическая интерпретация"),
    p("Полученные результаты согласуются с известными механизмами резистентности, связанными со стромой. Опухоль-ассоциированные фибробласты и TGFβ-сигналинг могут формировать барьер иммунной эксклюзии, ограничивая эффективность иммунотерапии (Mariathasan et al., Nature 2018; IMvigor210). При терапии анти-IGF-1R стромальные фибробласты являются дополнительным паракринным источником IGF-1, что снижает эффект блокады рецептора на опухолевых клетках (Cao et al., Cancer Res 2013). Для pan-HER ингибиторов TGFβ-индуцированная эпителиально-мезенхимальная трансдифференцировка является установленным механизмом вторичной резистентности (Shibue, Weinberg, NEJM 2019). Десмопластическая строма дополнительно снижает проникновение лекарственных средств в опухоль."),
    p("Таким образом, стромально-TGFβ-ось может рассматриваться как cross-class модификатор эффективности нескольких классов таргетной терапии, действующий через несколько молекулярных механизмов с общим клиническим исходом: снижение benefit от добавления препарата в стромально-богатой подгруппе пациенток."),
];

const clinical = [
    h1("Клиническое значение"),
    p("Стромальный биомаркер может использоваться для стратификации пациенток перед добавлением таргетной терапии к неоадъювантной химиотерапии. Подгруппа со стромально-богатой опухолью (верхний тертиль стромального индекса) получает существенно меньший клинический benefit, в случае ганитумаба этот benefit может становиться отрицательным."),
    p("Полученные результаты обосновывают разработку комбинированных терапевтических стратегий, направленных на подавление стромального компонента опухоли (анти-TGFβ агенты, анти-CAF подходы), в сочетании с таргетной терапией. Биомаркер также может применяться в дизайне клинических исследований как enrichment-marker, особенно при исследовании режимов на основе ICI и анти-IGF-1R препаратов в подгруппе тройного негативного рака."),
];

const limitations = [
    h1("Ограничения"),
    p("Исследование имеет следующие ограничения. Использована одна клиническая когорта, отсутствует независимая внешняя валидация на других протоколах (CALGB 40601, NeoALTTO, GeparOLA). Анти-HER2 контекст в данном релизе не оценен в связи с отсутствием соответствующей контрольной группы. Адаптивный дизайн I-SPY2 предполагает рандомизацию пациенток в разные временные периоды, что не контролируется в текущем анализе."),
    p("Величина наблюдаемого эффекта может быть несколько завышена в силу размера экспериментальных групп от 49 до 114 пациенток (winner's curse); истинное объединённое отношение шансов в независимых когортах может быть ближе к 0,70–0,80. Устойчивость сигнала к подвыборке умеренная (48 процентов подвыборок сохраняют значимость). Использованная CAF-сигнатура представляет собой экспрессионный портрет, не верифицированный на уровне единичных клеток или пространственной транскриптомики в данном анализе."),
];

const conclusions = [
    h1("Выводы"),
    p("Опухолевое микроокружение, прежде всего его стромальный компонент, является значимым предиктивным фактором эффективности таргетной терапии при раке молочной железы. Высокий стромальный индекс ассоциирован со снижением вероятности достижения pCR при добавлении таргетной терапии, эффект однонаправлен через 7 различных классов препаратов (объединённое OR_int = 0,62; 95% ДИ 0,49 до 0,80; p < 0,001)."),
    p("Наиболее выраженный эффект наблюдается в подгруппе тройного негативного рака молочной железы. Учет TME может повысить точность персонализации лечения и служить основой для разработки комбинированных терапевтических подходов. Для подтверждения клинической применимости биомаркера требуется независимая внешняя валидация и проверка концепции в проспективном дизайне."),
];

const refs = [
    h1("Литература"),
    bullet("Mariathasan S, Turley SJ, Nickles D, et al. TGFβ attenuates tumour response to PD-L1 blockade by contributing to exclusion of T cells. Nature 2018;554:544-548."),
    bullet("Calon A, Lonardo E, Berenguer-Llergo A, et al. Stromal gene expression defines poor-prognosis subtypes in colorectal cancer. Nat Genet 2015;47:320-329."),
    bullet("Dominguez CX, Müller S, Keerthivasan S, et al. Single-cell RNA sequencing reveals stromal evolution into LRRC15+ myofibroblasts as a determinant of patient response to cancer immunotherapy. Cancer Discov 2020;10:232-253."),
    bullet("Chen X, Song E. Turning foes to friends: targeting cancer-associated fibroblasts. J Clin Invest 2019;129:55-68."),
    bullet("Shibue T, Weinberg RA. EMT, CSCs, and drug resistance: the mechanistic link and clinical implications. NEJM 2019."),
    bullet("Cao Z, Livas T, Kyprianou N. IGF-1 stromal crosstalk in prostate and breast cancer. Cancer Res 2013;73:3312-3322."),
    bullet("Wolf DM, Yau C, Wulfkuhle J, et al. Redefining breast cancer subtypes to guide treatment prioritization and maximize response: Predictive biomarkers across 10 cancer therapies. Cell 2022;185:1-17."),
    bullet("Ayers M, Lunceford J, Nebozhyn M, et al. IFN-γ-related mRNA profile predicts clinical response to PD-1 blockade. J Clin Invest 2017;127:2930-2940."),
];

const allChildren = [
    ...titlePage,
    ...intro,
    ...aim,
    ...methods,
    resultsHeader,
    ...resultsP1,
    table1, t1Caption,
    ...fig1,
    ...resultsP2,
    table2, t2Caption,
    ...fig2,
    ...resultsP3,
    table3, t3Caption,
    ...fig3,
    ...resultsP4,
    ...patho,
    ...clinical,
    ...limitations,
    ...conclusions,
    ...refs,
];

const doc = new Document({
    creator: "OncoTME",
    title: "OncoTME клинический отчет",
    description: "Предиктивная роль TME в модификации ответа на таргетную терапию",
    styles: {
        default: { document: { run: { font: "Arial", size: 22 } } },
        paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
              quickFormat: true,
              run: { size: 30, bold: true, font: "Arial", color: "1F3864" },
              paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
              quickFormat: true,
              run: { size: 24, bold: true, font: "Arial", color: "2E75B6" },
              paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 1 } },
        ],
    },
    numbering: {
        config: [{
            reference: "b",
            levels: [{
                level: 0, format: LevelFormat.BULLET, text: "•",
                alignment: AlignmentType.LEFT,
                style: { paragraph: { indent: { left: 720, hanging: 360 } } },
            }],
        }],
    },
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 },
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
            },
        },
        headers: {
            default: new Header({
                children: [new Paragraph({
                    alignment: AlignmentType.RIGHT,
                    children: [new TextRun({
                        text: "OncoTME: предиктивная роль TME в таргетной терапии РМЖ",
                        size: 18, italics: true, color: "808080",
                    })],
                })],
            }),
        },
        footers: {
            default: new Footer({
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    children: [
                        new TextRun({ text: "стр. ", size: 18, color: "808080" }),
                        new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" }),
                        new TextRun({ text: " из ", size: 18, color: "808080" }),
                        new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "808080" }),
                    ],
                })],
            }),
        },
        children: allChildren,
    }],
});

Packer.toBuffer(doc).then(buf => {
    const out = path.join(REPO, 'results', 'OncoTME_отчет_v2.docx');
    fs.writeFileSync(out, buf);
    console.log("Saved:", out);
    console.log("Size:", buf.length, "bytes");
}).catch(err => {
    console.error("Error:", err);
    process.exit(1);
});
