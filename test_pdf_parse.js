import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse');
console.log(typeof pdfParse);
console.log(Object.keys(pdfParse));

import * as p from 'pdf-parse';
console.log(typeof p, Object.keys(p));
