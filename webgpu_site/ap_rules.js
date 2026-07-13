// AP Stylebook checker — JS port of ap_rules.py (runs live in-browser, no backend).
// Conservative: flags violations, avoids false-positives on correct text.
const NUM_WORDS={"1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine"};
const ORD_WORDS={"1st":"first","2nd":"second","3rd":"third","4th":"fourth","5th":"fifth","6th":"sixth","7th":"seventh","8th":"eighth","9th":"ninth"};
const ABBR_MONTHS={January:"Jan.",February:"Feb.",August:"Aug.",September:"Sept.",October:"Oct.",November:"Nov.",December:"Dec."};
const ABBR_STATES={California:"Calif.",Florida:"Fla.",Pennsylvania:"Pa.",Massachusetts:"Mass.",Illinois:"Ill.",Georgia:"Ga.",Arizona:"Ariz.",Colorado:"Colo.",Michigan:"Mich.",Missouri:"Mo.",Connecticut:"Conn.",Oregon:"Ore.",Kentucky:"Ky.",Tennessee:"Tenn.",Virginia:"Va."};
const ONES={one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9};
const TENS={ten:10,eleven:11,twelve:12,thirteen:13,fourteen:14,fifteen:15,sixteen:16,seventeen:17,eighteen:18,nineteen:19,twenty:20,thirty:30,forty:40,fifty:50,sixty:60,seventy:70,eighty:80,ninety:90};

function wordToNum(w){
  w=w.toLowerCase().trim();
  if(w in TENS)return TENS[w];
  if(w in ONES)return ONES[w];
  if(w.includes("-")){const[a,b]=w.split("-");if(TENS[a]>=20&&b in ONES)return TENS[a]+ONES[b];}
  return null;
}

const SUBS=[
  [/\be-mail\b/g,"spelling: AP uses “email” (no hyphen)","email"],
  [/\bweb site\b/gi,"spelling: AP uses “website” (one word)","website"],
  [/\bunder way\b/g,"spelling: AP uses “underway” (one word)","underway"],
  [/\bokay\b/g,"usage: AP uses “OK”, not “okay”","OK"],
  [/\badvisor\b/g,"spelling: AP uses “adviser”","adviser"],
  [/\btowards\b/g,"usage: AP uses “toward” (no s)","toward"],
  [/\bbackwards\b/g,"usage: AP uses “backward” (no s)","backward"],
  [/\bInternet\b/g,"capitalization: AP lowercases “internet”","internet"],
  [/\bversus\b/g,"usage: AP abbreviates to “vs.”","vs."],
];

function apCheck(text){
  const hits=[]; const seen=new Set();
  const add=(span,rule,sug)=>{const k=span+"|"+rule; if(!seen.has(k)){seen.add(k);hits.push({span,rule,suggestion:sug});}};
  let m;

  // 1) numbers 1-9 as figures (skip %, times, ordinals, dates, ages, money)
  const numRe=/(?<![\d$%.:'-])\b([1-9])\b(?![%'\d-])/g;
  while((m=numRe.exec(text))){
    const after=text.slice(m.index+1,m.index+14), before=text.slice(Math.max(0,m.index-14),m.index);
    if(/^\s*(?:%|percent|:|st|nd|rd|th|a\.m|p\.m|\s*[AaPp]\.?[Mm]|-year|\s+years?\s+old)/.test(after))continue;
    if(/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+$/.test(before))continue;
    if(before.trimEnd().endsWith("$"))continue;
    add(m[1],"numbers: spell out one through nine; figures for 10+",`replace “${m[1]}” with “${NUM_WORDS[m[1]]}”`);
  }
  // 2) ordinals first-ninth
  const ordRe=/\b([1-9](?:st|nd|rd|th))\b/g;
  while((m=ordRe.exec(text))){if(ORD_WORDS[m[1]])add(m[1],"ordinals: spell out first through ninth",`“${ORD_WORDS[m[1]]}”`);}
  // 2b) ages -> figures
  const ageRe=/\b([A-Za-z]+(?:-[A-Za-z]+)?)-year-old\b/g;
  while((m=ageRe.exec(text))){const n=wordToNum(m[1]);if(n!==null)add(m[0],"ages: use figures for ages (hyphenate as a modifier)",`“${n}-year-old”`);}
  const ageRe2=/\b([a-z]+(?:-[a-z]+)?)\s+years?\s+old\b/g;
  while((m=ageRe2.exec(text))){const n=wordToNum(m[1]);if(n!==null)add(m[0],"ages: use figures for ages",`“${n} years old”`);}
  // 3) time: uppercase AM/PM or :00, and 12 a.m./p.m.
  const timeRe=/\b(\d{1,2})(:\d{2})?\s*([APap]\.?[Mm]\.?)/g;
  while((m=timeRe.exec(text))){
    const hour=m[1],mins=m[2],ap=m[3],norm=ap.toUpperCase().startsWith("A")?"a.m.":"p.m.";
    if(hour==="12"&&(!mins||mins===":00")){add(m[0].trim(),"time: use “noon”/“midnight”, not 12 p.m./12 a.m.",norm==="p.m."?"noon":"midnight");continue;}
    if(ap!=="a.m."&&ap!=="p.m."||mins===":00"){
      const fixed=hour+((!mins||mins===":00")?"":mins)+" "+norm;
      add(m[0].trim(),"time: lowercase a.m./p.m. with periods; drop “:00” on the hour",`use “${fixed}”`);
    }
  }
  // 4) months abbreviated with a date
  for(const[full,abbr]of Object.entries(ABBR_MONTHS)){
    const r=new RegExp("\\b"+full+"\\s+(\\d{1,2})\\b","g");
    while((m=r.exec(text)))add(m[0],"months: abbreviate Jan./Feb./Aug./Sept./Oct./Nov./Dec. with a date",`“${abbr} ${m[1]}”`);
  }
  // 5) states abbreviated with a city
  for(const[full,abbr]of Object.entries(ABBR_STATES)){
    const r=new RegExp("\\b([A-Z][a-zA-Z]+),\\s+"+full+"\\b","g");
    while((m=r.exec(text)))add(m[0],`states: AP abbreviates ${full} as ${abbr} with a city`,`“${m[1]}, ${abbr}”`);
  }
  // 6) Oxford comma
  const oxRe=/(\w+),\s+(\w+),\s+and\s+(\w+)/g;
  while((m=oxRe.exec(text)))add(m[0],"punctuation: no serial (Oxford) comma in a simple series",m[0].replace(m[2]+", and",m[2]+" and"));
  // 7) percent -> %
  const pctRe=/\b(\d+)\s+percent\b/g;
  while((m=pctRe.exec(text)))add(m[0],"percent: use the % sign with a numeral (AP, 2019+)",`“${m[1]}%”`);
  // 8) over N -> more than N
  const overRe=/\bover\s+(\d[\d,]*)\b/g;
  while((m=overRe.exec(text)))add(m[0],"usage: use “more than” (not “over”) with a numeral quantity",m[0].replace("over","more than"));
  // 9) attribution: said <bare Name>. -> Name said
  const saidRe=/\bsaid\s+([A-Z][a-z]+)\s*([.!?"'”’]|$)/g;
  while((m=saidRe.exec(text)))add(m[0].trim(),"attribution: name before “said” unless a title/clause follows the name",`“${m[1]} said”`);
  // 10) decade apostrophe
  const decRe=/\b(\d{2,4})'s\b/g;
  while((m=decRe.exec(text)))add(m[0],"decades: no apostrophe before the s (1990s)",`“${m[1]}s”`);
  // 11) money words -> $
  const moneyRe=/\b(\d[\d,]*)\s+dollars?\b/g;
  while((m=moneyRe.exec(text)))add(m[0],"money: use the $ sign with figures",`“$${m[1]}”`);
  // 13) simple subs
  for(const[re,rule,repl]of SUBS){
    const r=new RegExp(re.source,re.flags);
    while((m=r.exec(text))){const fixed=m[0].replace(new RegExp(re.source,re.flags.replace("g","")),repl);if(fixed!==m[0])add(m[0],rule,`“${fixed}”`);}
  }
  return hits;
}
