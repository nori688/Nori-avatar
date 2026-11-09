import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "/home/nori/Проекты/Nori/outputs/text_avatar_dataset";

const intents = [
  {
    name: "neutral",
    phrases: [
      "стой спокойно", "нейтральное выражение", "смотри прямо",
      "вернись в обычное состояние", "ничего не изображай",
      "спокойное лицо", "без эмоций", "обычная поза",
      "прими нейтральный вид", "расслабь лицо", "будь спокоен",
      "верни лицо в центр",
    ],
    target: {},
  },
  {
    name: "happy",
    phrases: [
      "улыбнись", "я очень рад", "покажи радость", "мне весело",
      "это отличная новость", "выгляди счастливым", "давай улыбку",
      "я в хорошем настроении", "обрадуйся", "покажи счастливое лицо",
      "как же здорово", "я счастлив тебя видеть",
    ],
    target: { left_brow_y: -0.35, right_brow_y: -0.35, mouth_open: 0.45, face_angle: 0.08 },
  },
  {
    name: "sad",
    phrases: [
      "мне грустно", "покажи печаль", "это плохая новость",
      "выгляди расстроенным", "мне сейчас невесело", "опусти взгляд от печали",
      "изобрази грусть", "я сильно расстроен", "печальное лицо",
      "настроение совсем упало", "мне хочется плакать", "как же грустно",
    ],
    target: { left_brow_y: 0.45, right_brow_y: 0.45, left_eye_y: 0.35, right_eye_y: 0.35, mouth_open: 0.12, face_angle: -0.08 },
  },
  {
    name: "angry",
    phrases: [
      "я злюсь", "покажи злость", "сердитое лицо", "это меня раздражает",
      "нахмурься", "выгляди сердитым", "я очень недоволен",
      "изобрази гнев", "меня это бесит", "разозлись",
      "сделай грозный взгляд", "я в ярости",
    ],
    target: { left_brow_y: 0.55, right_brow_y: 0.55, left_eye_y: 0.15, right_eye_y: 0.15, mouth_x: -0.18, mouth_open: 0.18, face_angle: -0.1 },
  },
  {
    name: "surprised",
    phrases: [
      "я удивлён", "покажи удивление", "ничего себе", "вот это да",
      "сделай удивлённое лицо", "это неожиданно", "удивись",
      "я не могу поверить", "какой сюрприз", "сильно удивись",
      "глаза шире", "я в полном шоке",
    ],
    target: { left_brow_y: -0.7, right_brow_y: -0.7, mouth_open: 0.9, face_angle: 0.04 },
  },
  {
    name: "look_left",
    phrases: [
      "посмотри налево", "смотри влево", "переведи взгляд влево",
      "глаза налево", "взгляни в левую сторону", "посмотри левее",
      "поверни взгляд налево", "следи за чем-то слева",
      "обрати внимание налево", "глянь влево", "смотри далеко влево",
      "резко посмотри налево",
    ],
    target: { left_eye_x: -0.9, right_eye_x: -0.9, face_angle: -0.25 },
  },
  {
    name: "look_right",
    phrases: [
      "посмотри направо", "смотри вправо", "переведи взгляд вправо",
      "глаза направо", "взгляни в правую сторону", "посмотри правее",
      "поверни взгляд направо", "следи за чем-то справа",
      "обрати внимание направо", "глянь вправо", "смотри далеко вправо",
      "резко посмотри направо",
    ],
    target: { left_eye_x: 0.9, right_eye_x: 0.9, face_angle: 0.25 },
  },
  {
    name: "look_up",
    phrases: [
      "посмотри вверх", "смотри наверх", "подними взгляд",
      "глаза вверх", "взгляни на потолок", "посмотри выше",
      "переведи глаза наверх", "обрати внимание наверх",
      "глянь вверх", "смотри высоко", "сильно подними взгляд",
      "быстро посмотри вверх",
    ],
    target: { left_eye_y: -0.9, right_eye_y: -0.9, left_brow_y: -0.15, right_brow_y: -0.15 },
  },
  {
    name: "look_down",
    phrases: [
      "посмотри вниз", "смотри вниз", "опусти взгляд", "глаза вниз",
      "взгляни под ноги", "посмотри ниже", "переведи глаза вниз",
      "обрати внимание вниз", "глянь вниз", "смотри совсем низко",
      "сильно опусти взгляд", "быстро посмотри вниз",
    ],
    target: { left_eye_y: 0.9, right_eye_y: 0.9, left_brow_y: 0.12, right_brow_y: 0.12 },
  },
  {
    name: "blink",
    phrases: [
      "моргни", "закрой глаза", "быстро моргни", "прикрой глаза",
      "сделай моргание", "сомкни веки", "закрой оба глаза",
      "моргни один раз", "зажмурься", "прикрой оба глаза",
      "сильно зажмурься", "полностью закрой глаза",
    ],
    target: { left_eye_open: 0.0, right_eye_open: 0.0 },
  },
  {
    name: "tilt_left",
    phrases: [
      "наклони голову налево", "наклонись влево", "голову влево",
      "сделай наклон влево", "поверни голову немного налево",
      "склони голову в левую сторону", "качни голову влево",
      "держи голову левее", "наклони лицо налево", "голова сильно влево",
      "сильнее наклонись влево", "максимальный наклон налево",
    ],
    target: { face_angle: -0.85 },
  },
  {
    name: "tilt_right",
    phrases: [
      "наклони голову направо", "наклонись вправо", "голову вправо",
      "сделай наклон вправо", "поверни голову немного направо",
      "склони голову в правую сторону", "качни голову вправо",
      "держи голову правее", "наклони лицо направо", "голова сильно вправо",
      "сильнее наклонись вправо", "максимальный наклон направо",
    ],
    target: { face_angle: 0.85 },
  },
  {
    name: "speak",
    phrases: [
      "скажи что-нибудь", "начни говорить", "произнеси фразу",
      "открой рот", "покажи разговор", "говори", "ответь мне",
      "скажи это вслух", "изобрази речь", "говори активнее",
      "скажи громко", "открой рот шире",
    ],
    target: { mouth_open: 0.8, mouth_x: 0.08 },
  },
];

const outputNames = [
  "left_brow_y", "right_brow_y",
  "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
  "left_eye_open", "right_eye_open",
  "mouth_x", "mouth_open", "face_angle",
];

const defaultOutput = {
  left_brow_y: 0, right_brow_y: 0,
  left_eye_x: 0, left_eye_y: 0, right_eye_x: 0, right_eye_y: 0,
  left_eye_open: 1, right_eye_open: 1,
  mouth_x: 0, mouth_open: 0.08, face_angle: 0,
};

function splitForPhrase(index) {
  if (index === 10) return "validation";
  if (index === 11) return "test";
  return "train";
}

function intensityForPhrase(index) {
  return [0.5, 0.6, 0.7, 0.8, 0.9, 1.0][index % 6];
}

function textVariants(text) {
  return [
    text,
    `пожалуйста, ${text}`,
    `${text}!`,
    text.charAt(0).toUpperCase() + text.slice(1),
  ];
}

function makeTarget(intent, intensity) {
  const result = { ...defaultOutput };
  for (const [name, value] of Object.entries(intent.target)) {
    if (name === "left_eye_open" || name === "right_eye_open") {
      result[name] = 1 - intensity;
    } else if (name === "mouth_open") {
      result[name] = Math.max(0.08, value * intensity);
    } else {
      result[name] = value * intensity;
    }
  }
  return result;
}

const rows = [];
let id = 1;
for (const intent of intents) {
  intent.phrases.forEach((phrase, phraseIndex) => {
    const intensity = intensityForPhrase(phraseIndex);
    const target = makeTarget(intent, intensity);
    for (const text of textVariants(phrase)) {
      rows.push([
        id,
        splitForPhrase(phraseIndex),
        text,
        intent.name,
        intensity,
        ...outputNames.map((name) => Number(target[name].toFixed(4))),
      ]);
      id += 1;
    }
  });
}

const headers = ["id", "split", "text", "intent", "intensity", ...outputNames];

function csvCell(value) {
  const text = String(value);
  if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

const csv = [headers, ...rows]
  .map((row) => row.map(csvCell).join(","))
  .join("\n") + "\n";

await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(`${outputDir}/avatar_text_dataset.csv`, csv, "utf8");

const workbook = Workbook.create();
const dataset = workbook.worksheets.add("Dataset");
const schema = workbook.worksheets.add("Schema");

dataset.showGridLines = false;
dataset.getRange("A1:P1").merge();
dataset.getRange("A1").values = [["Датасет: русский текст → параметры 2D-аватара"]];
dataset.getRange("A1").format = {
  font: { name: "Arial", size: 14, bold: true, color: "#1F2937" },
};
dataset.getRange("A3:P3").values = [headers];
dataset.getRange("A4").write(rows);
dataset.getRange(`A3:P${rows.length + 3}`).format.font = { name: "Arial", size: 10 };
dataset.getRange("A3:P3").format = {
  fill: "#334155",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
dataset.getRange(`E4:P${rows.length + 3}`).format.numberFormat = "0.0000";
dataset.getRange(`A3:P${rows.length + 3}`).format.verticalAlignment = "center";
dataset.getRange(`A3:P${rows.length + 3}`).format.autofitColumns();
dataset.getRange("A:A").format.columnWidth = 8;
dataset.getRange("B:B").format.columnWidth = 13;
dataset.getRange("C:C").format.columnWidth = 44;
dataset.getRange("D:D").format.columnWidth = 15;
dataset.getRange("E:P").format.columnWidth = 15;
dataset.freezePanes.freezeRows(3);
dataset.tables.add(`A3:P${rows.length + 3}`, true, "AvatarDatasetTable").style = "TableStyleMedium2";

const schemaRows = [
  ["id", "целое число", "Уникальный номер примера"],
  ["split", "train / validation / test", "Готовое разделение данных"],
  ["text", "строка", "Русская команда или эмоциональная реплика; это вход модели"],
  ["intent", "строка", "Класс смысла текста; удобен для проверки и простой классификации"],
  ["intensity", "0…1", "Сила выражения или движения"],
  ["left_brow_y", "-1…1", "Смещение левой брови: минус вверх, плюс вниз"],
  ["right_brow_y", "-1…1", "Смещение правой брови: минус вверх, плюс вниз"],
  ["left_eye_x", "-1…1", "Горизонтальное смещение левого глаза"],
  ["left_eye_y", "-1…1", "Вертикальное смещение левого глаза"],
  ["right_eye_x", "-1…1", "Горизонтальное смещение правого глаза"],
  ["right_eye_y", "-1…1", "Вертикальное смещение правого глаза"],
  ["left_eye_open", "0…1", "Открытость левого глаза: 0 закрыт, 1 открыт"],
  ["right_eye_open", "0…1", "Открытость правого глаза: 0 закрыт, 1 открыт"],
  ["mouth_x", "-1…1", "Горизонтальное смещение рта"],
  ["mouth_open", "0…1", "Открытость рта"],
  ["face_angle", "-1…1", "Нормализованный наклон головы"],
];

schema.showGridLines = false;
schema.getRange("A1:C1").merge();
schema.getRange("A1").values = [["Описание колонок"]];
schema.getRange("A1").format = {
  font: { name: "Arial", size: 14, bold: true, color: "#1F2937" },
};
schema.getRange("A3:C3").values = [["Колонка", "Диапазон / тип", "Назначение"]];
schema.getRange("A4").write(schemaRows);
schema.getRange("A3:C3").format = {
  fill: "#334155",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
};
schema.getRange(`A3:C${schemaRows.length + 3}`).format.font = { name: "Arial", size: 10 };
schema.getRange(`A3:C${schemaRows.length + 3}`).format.verticalAlignment = "center";
schema.getRange("A:A").format.columnWidth = 22;
schema.getRange("B:B").format.columnWidth = 25;
schema.getRange("C:C").format.columnWidth = 72;
schema.freezePanes.freezeRows(3);
schema.tables.add(`A3:C${schemaRows.length + 3}`, true, "SchemaTable").style = "TableStyleMedium2";

workbook.recalculate();

const check = await workbook.inspect({
  kind: "table",
  range: "Dataset!A1:P12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 16,
});
console.log(check.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Dataset",
  range: "A1:P12",
  scale: 1,
  format: "png",
});
await fs.writeFile(`${outputDir}/preview.png`, new Uint8Array(await preview.arrayBuffer()));

const schemaPreview = await workbook.render({
  sheetName: "Schema",
  range: "A1:C19",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  `${outputDir}/schema_preview.png`,
  new Uint8Array(await schemaPreview.arrayBuffer()),
);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(`${outputDir}/avatar_text_dataset.xlsx`);

console.log(JSON.stringify({ rows: rows.length, outputDir }));
