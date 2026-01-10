// Gardiner Sign List - Common hieroglyphs with their meanings
// This is a subset of the full Gardiner classification system

export const gardinerSigns = {
  // A - Man and his occupations
  'A1': { name: 'Seated Man', meaning: 'man, person', phonetic: 'i', category: 'A - Man and his occupations' },
  'A2': { name: 'Man with hand to mouth', meaning: 'eat, drink, speak', phonetic: '', category: 'A - Man and his occupations' },
  'A3': { name: 'Man sitting on heel', meaning: 'sit', phonetic: '', category: 'A - Man and his occupations' },
  'A4': { name: 'Seated man with arms raised', meaning: 'jubilation, praise', phonetic: '', category: 'A - Man and his occupations' },
  'A5': { name: 'Man hiding behind wall', meaning: 'hide', phonetic: '', category: 'A - Man and his occupations' },
  'A14': { name: 'Falling man', meaning: 'fall, enemy', phonetic: '', category: 'A - Man and his occupations' },
  'A17': { name: 'Child sitting', meaning: 'child, young', phonetic: '', category: 'A - Man and his occupations' },
  'A24': { name: 'Man striking', meaning: 'strike, force', phonetic: '', category: 'A - Man and his occupations' },
  'A26': { name: 'Man calling', meaning: 'call, beckon', phonetic: '', category: 'A - Man and his occupations' },
  'A28': { name: 'Man with arms raised high', meaning: 'high, exalt', phonetic: '', category: 'A - Man and his occupations' },
  'A40': { name: 'Seated god', meaning: 'god, king', phonetic: '', category: 'A - Man and his occupations' },
  
  // B - Woman and her occupations
  'B1': { name: 'Seated Woman', meaning: 'woman', phonetic: '', category: 'B - Woman' },
  'B2': { name: 'Pregnant woman', meaning: 'pregnant, conceive', phonetic: '', category: 'B - Woman' },
  'B3': { name: 'Woman giving birth', meaning: 'give birth', phonetic: '', category: 'B - Woman' },
  
  // C - Anthropomorphic deities
  'C1': { name: 'Re with sun disk', meaning: 'Ra, sun god', phonetic: '', category: 'C - Deities' },
  'C2': { name: 'Horus', meaning: 'Horus', phonetic: '', category: 'C - Deities' },
  'C3': { name: 'Thoth with ibis head', meaning: 'Thoth', phonetic: '', category: 'C - Deities' },
  'C4': { name: 'Khnum with ram head', meaning: 'Khnum', phonetic: '', category: 'C - Deities' },
  'C6': { name: 'Anubis', meaning: 'Anubis', phonetic: '', category: 'C - Deities' },
  'C7': { name: 'Seth', meaning: 'Seth', phonetic: '', category: 'C - Deities' },
  'C10': { name: 'Maat with feather', meaning: 'Maat, truth', phonetic: '', category: 'C - Deities' },
  'C11': { name: 'Heh', meaning: 'million, eternity', phonetic: '', category: 'C - Deities' },
  
  // D - Parts of the human body
  'D1': { name: 'Head in profile', meaning: 'head', phonetic: 'tp', category: 'D - Body parts' },
  'D2': { name: 'Face', meaning: 'face', phonetic: 'ḥr', category: 'D - Body parts' },
  'D3': { name: 'Hair', meaning: 'hair', phonetic: '', category: 'D - Body parts' },
  'D4': { name: 'Eye', meaning: 'eye, see', phonetic: 'ir', category: 'D - Body parts' },
  'D5': { name: 'Eye with cosmetic', meaning: 'eye makeup', phonetic: '', category: 'D - Body parts' },
  'D6': { name: 'Eye with marking', meaning: 'sight', phonetic: '', category: 'D - Body parts' },
  'D10': { name: 'Eye of Horus', meaning: 'Eye of Horus, protection', phonetic: 'wḏꜣt', category: 'D - Body parts' },
  'D19': { name: 'Nose and mouth', meaning: 'nose, joy', phonetic: 'fnd', category: 'D - Body parts' },
  'D21': { name: 'Mouth', meaning: 'mouth', phonetic: 'r', category: 'D - Body parts' },
  'D28': { name: 'Two arms', meaning: 'ka, soul', phonetic: 'kꜣ', category: 'D - Body parts' },
  'D36': { name: 'Arm', meaning: 'arm', phonetic: 'ꜥ', category: 'D - Body parts' },
  'D46': { name: 'Hand', meaning: 'hand', phonetic: 'd', category: 'D - Body parts' },
  'D54': { name: 'Walking legs', meaning: 'walk, run', phonetic: '', category: 'D - Body parts' },
  'D58': { name: 'Foot', meaning: 'foot', phonetic: 'b', category: 'D - Body parts' },
  
  // E - Mammals
  'E1': { name: 'Bull', meaning: 'bull', phonetic: 'kꜣ', category: 'E - Mammals' },
  'E6': { name: 'Horse', meaning: 'horse', phonetic: '', category: 'E - Mammals' },
  'E7': { name: 'Donkey', meaning: 'donkey', phonetic: '', category: 'E - Mammals' },
  'E8': { name: 'Kid', meaning: 'kid, child', phonetic: '', category: 'E - Mammals' },
  'E9': { name: 'Newborn calf', meaning: 'calf', phonetic: '', category: 'E - Mammals' },
  'E10': { name: 'Ram', meaning: 'ram', phonetic: '', category: 'E - Mammals' },
  'E14': { name: 'Dog', meaning: 'dog', phonetic: '', category: 'E - Mammals' },
  'E15': { name: 'Lying dog', meaning: 'jackal', phonetic: '', category: 'E - Mammals' },
  'E17': { name: 'Jackal', meaning: 'Anubis', phonetic: 'zꜣb', category: 'E - Mammals' },
  'E20': { name: 'Set animal', meaning: 'Seth', phonetic: '', category: 'E - Mammals' },
  'E21': { name: 'Lying down Seth animal', meaning: 'Seth', phonetic: '', category: 'E - Mammals' },
  'E23': { name: 'Lion lying down', meaning: 'lion', phonetic: '', category: 'E - Mammals' },
  'E24': { name: 'Panther', meaning: 'fierce', phonetic: '', category: 'E - Mammals' },
  'E26': { name: 'Elephant', meaning: 'elephant', phonetic: 'ꜣbw', category: 'E - Mammals' },
  'E34': { name: 'Hare', meaning: 'hare', phonetic: 'wn', category: 'E - Mammals' },
  
  // F - Parts of mammals
  'F1': { name: 'Ox head', meaning: 'ox', phonetic: '', category: 'F - Parts of mammals' },
  'F4': { name: 'Front of lion', meaning: 'front, beginning', phonetic: 'ḥꜣt', category: 'F - Parts of mammals' },
  'F12': { name: 'Head and neck of jackal', meaning: 'neck', phonetic: '', category: 'F - Parts of mammals' },
  'F13': { name: 'Horns', meaning: 'horns', phonetic: 'wp', category: 'F - Parts of mammals' },
  'F18': { name: 'Tusk of elephant', meaning: 'tusk', phonetic: 'ḥw', category: 'F - Parts of mammals' },
  'F20': { name: 'Tongue', meaning: 'tongue', phonetic: 'ns', category: 'F - Parts of mammals' },
  'F21': { name: 'Ear of bovine', meaning: 'ear', phonetic: 'sḏm', category: 'F - Parts of mammals' },
  'F22': { name: 'Hind quarters', meaning: 'hind', phonetic: 'pḥ', category: 'F - Parts of mammals' },
  'F26': { name: 'Skin of animal', meaning: 'skin', phonetic: '', category: 'F - Parts of mammals' },
  'F29': { name: 'Cow skin with arrow', meaning: 'pierce', phonetic: 'sṯi', category: 'F - Parts of mammals' },
  'F31': { name: 'Three skins', meaning: 'skin', phonetic: 'ms', category: 'F - Parts of mammals' },
  'F32': { name: 'Animal belly', meaning: 'belly', phonetic: '', category: 'F - Parts of mammals' },
  'F34': { name: 'Heart', meaning: 'heart', phonetic: 'ib', category: 'F - Parts of mammals' },
  'F35': { name: 'Heart and windpipe', meaning: 'good, beautiful', phonetic: 'nfr', category: 'F - Parts of mammals' },
  'F40': { name: 'Backbone and ribs', meaning: 'backbone', phonetic: 'ꜣw', category: 'F - Parts of mammals' },
  'F44': { name: 'Bone with meat', meaning: 'bone', phonetic: 'qs', category: 'F - Parts of mammals' },
  
  // G - Birds
  'G1': { name: 'Egyptian vulture', meaning: 'vulture', phonetic: 'ꜣ', category: 'G - Birds' },
  'G4': { name: 'Long legged buzzard', meaning: 'buzzard', phonetic: 'tyw', category: 'G - Birds' },
  'G5': { name: 'Falcon', meaning: 'Horus', phonetic: '', category: 'G - Birds' },
  'G7': { name: 'Falcon on standard', meaning: 'god', phonetic: '', category: 'G - Birds' },
  'G14': { name: 'Vulture', meaning: 'vulture, mother', phonetic: 'mwt', category: 'G - Birds' },
  'G17': { name: 'Owl', meaning: 'owl', phonetic: 'm', category: 'G - Birds' },
  'G21': { name: 'Guinea fowl', meaning: 'guinea fowl', phonetic: 'nḥ', category: 'G - Birds' },
  'G25': { name: 'Crested ibis', meaning: 'ibis', phonetic: 'ꜣḫ', category: 'G - Birds' },
  'G26': { name: 'Sacred ibis', meaning: 'Thoth', phonetic: '', category: 'G - Birds' },
  'G29': { name: 'Jabiru', meaning: 'soul', phonetic: 'bꜣ', category: 'G - Birds' },
  'G35': { name: 'Cormorant', meaning: 'great', phonetic: 'ꜥꜣ', category: 'G - Birds' },
  'G36': { name: 'Swallow', meaning: 'great', phonetic: 'wr', category: 'G - Birds' },
  'G37': { name: 'Sparrow', meaning: 'small, bad', phonetic: '', category: 'G - Birds' },
  'G38': { name: 'White fronted goose', meaning: 'goose', phonetic: 'gb', category: 'G - Birds' },
  'G39': { name: 'Pintail duck', meaning: 'duck', phonetic: 'sꜣ', category: 'G - Birds' },
  'G40': { name: 'Duck flying', meaning: 'fly', phonetic: 'pꜣ', category: 'G - Birds' },
  'G43': { name: 'Quail chick', meaning: 'chick', phonetic: 'w', category: 'G - Birds' },
  'G47': { name: 'Duckling', meaning: 'duckling', phonetic: 'ṯꜣ', category: 'G - Birds' },
  
  // H - Parts of birds
  'H1': { name: 'Head of pintail', meaning: 'head', phonetic: '', category: 'H - Parts of birds' },
  'H2': { name: 'Head of crested bird', meaning: 'head', phonetic: '', category: 'H - Parts of birds' },
  'H6': { name: 'Feather', meaning: 'truth, Maat', phonetic: 'šw', category: 'H - Parts of birds' },
  'H8': { name: 'Egg', meaning: 'egg', phonetic: '', category: 'H - Parts of birds' },
  
  // I - Amphibious animals, reptiles
  'I1': { name: 'Gecko', meaning: 'many', phonetic: '', category: 'I - Reptiles' },
  'I3': { name: 'Crocodile', meaning: 'crocodile', phonetic: '', category: 'I - Reptiles' },
  'I5': { name: 'Crocodile with tail', meaning: 'aggression', phonetic: '', category: 'I - Reptiles' },
  'I6': { name: 'Crocodile scale', meaning: 'collect', phonetic: '', category: 'I - Reptiles' },
  'I9': { name: 'Horned viper', meaning: 'viper', phonetic: 'f', category: 'I - Reptiles' },
  'I10': { name: 'Cobra', meaning: 'goddess', phonetic: 'ḏ', category: 'I - Reptiles' },
  'I11': { name: 'Cobra on basket', meaning: 'goddess', phonetic: '', category: 'I - Reptiles' },
  'I12': { name: 'Erect cobra', meaning: 'goddess', phonetic: 'iꜥrt', category: 'I - Reptiles' },
  'I14': { name: 'Snake', meaning: 'snake', phonetic: '', category: 'I - Reptiles' },
  
  // K - Fish and parts of fish
  'K1': { name: 'Tilapia fish', meaning: 'tilapia', phonetic: 'in', category: 'K - Fish' },
  'K3': { name: 'Mullet', meaning: 'mullet', phonetic: 'ꜥḏ', category: 'K - Fish' },
  'K4': { name: 'Oxyrhynchus fish', meaning: 'oxyrynchus', phonetic: '', category: 'K - Fish' },
  'K5': { name: 'Petrocephalus fish', meaning: 'petrocephalus', phonetic: '', category: 'K - Fish' },
  
  // L - Invertebrates and smaller animals
  'L1': { name: 'Scarab beetle', meaning: 'scarab, become', phonetic: 'ḫpr', category: 'L - Invertebrates' },
  'L2': { name: 'Bee', meaning: 'bee, king of Lower Egypt', phonetic: 'bit', category: 'L - Invertebrates' },
  'L3': { name: 'Fly', meaning: 'fly', phonetic: '', category: 'L - Invertebrates' },
  'L7': { name: 'Scorpion', meaning: 'scorpion', phonetic: 'srqt', category: 'L - Invertebrates' },
  
  // M - Trees and plants
  'M1': { name: 'Tree', meaning: 'tree', phonetic: '', category: 'M - Trees and plants' },
  'M2': { name: 'Plant', meaning: 'plant', phonetic: 'ḥn', category: 'M - Trees and plants' },
  'M3': { name: 'Branch', meaning: 'wood', phonetic: 'ḫt', category: 'M - Trees and plants' },
  'M4': { name: 'Palm branch', meaning: 'year', phonetic: 'rnpt', category: 'M - Trees and plants' },
  'M8': { name: 'Pool with lotus', meaning: 'pool', phonetic: 'šꜣ', category: 'M - Trees and plants' },
  'M12': { name: 'Lotus', meaning: 'lotus', phonetic: '', category: 'M - Trees and plants' },
  'M13': { name: 'Papyrus plant', meaning: 'papyrus', phonetic: 'wꜣḏ', category: 'M - Trees and plants' },
  'M16': { name: 'Clump of papyrus', meaning: 'Lower Egypt', phonetic: 'ḥꜣ', category: 'M - Trees and plants' },
  'M17': { name: 'Reed', meaning: 'reed', phonetic: 'i', category: 'M - Trees and plants' },
  'M18': { name: 'Reed combination', meaning: 'reed', phonetic: 'ii', category: 'M - Trees and plants' },
  'M20': { name: 'Field of reeds', meaning: 'field', phonetic: '', category: 'M - Trees and plants' },
  'M22': { name: 'Rush', meaning: 'Upper Egypt', phonetic: 'šmꜥ', category: 'M - Trees and plants' },
  'M23': { name: 'Sedge plant', meaning: 'king of Upper Egypt', phonetic: 'sw', category: 'M - Trees and plants' },
  'M26': { name: 'Flowering sedge', meaning: 'Upper Egypt', phonetic: 'šmꜥw', category: 'M - Trees and plants' },
  'M29': { name: 'Pod', meaning: 'seed', phonetic: 'nḏm', category: 'M - Trees and plants' },
  'M40': { name: 'Bundle of reeds', meaning: 'bundle', phonetic: 'is', category: 'M - Trees and plants' },
  'M42': { name: 'Flower', meaning: 'flower', phonetic: 'wn', category: 'M - Trees and plants' },
  'M44': { name: 'Thorn', meaning: 'thorn', phonetic: 'spd', category: 'M - Trees and plants' },
  
  // N - Sky, earth, water
  'N1': { name: 'Sky', meaning: 'sky, heaven', phonetic: 'pt', category: 'N - Sky, earth, water' },
  'N5': { name: 'Sun', meaning: 'sun, day, Ra', phonetic: 'rꜥ', category: 'N - Sky, earth, water' },
  'N6': { name: 'Sun with uraeus', meaning: 'sun god', phonetic: '', category: 'N - Sky, earth, water' },
  'N8': { name: 'Sunshine', meaning: 'sunshine', phonetic: '', category: 'N - Sky, earth, water' },
  'N9': { name: 'Moon', meaning: 'moon, month', phonetic: 'iꜥḥ', category: 'N - Sky, earth, water' },
  'N11': { name: 'Crescent moon', meaning: 'moon', phonetic: '', category: 'N - Sky, earth, water' },
  'N14': { name: 'Star', meaning: 'star, hour', phonetic: 'sbꜣ', category: 'N - Sky, earth, water' },
  'N16': { name: 'Land with grains', meaning: 'land', phonetic: 'tꜣ', category: 'N - Sky, earth, water' },
  'N17': { name: 'Land', meaning: 'land', phonetic: 'tꜣ', category: 'N - Sky, earth, water' },
  'N18': { name: 'Sandy tract', meaning: 'island', phonetic: 'iw', category: 'N - Sky, earth, water' },
  'N21': { name: 'Tongue of land', meaning: 'tongue', phonetic: '', category: 'N - Sky, earth, water' },
  'N24': { name: 'Irrigation canal', meaning: 'irrigated land', phonetic: '', category: 'N - Sky, earth, water' },
  'N25': { name: 'Sandy hill', meaning: 'foreign land', phonetic: 'ḫꜣst', category: 'N - Sky, earth, water' },
  'N26': { name: 'Mountain', meaning: 'mountain', phonetic: 'ḏw', category: 'N - Sky, earth, water' },
  'N28': { name: 'Sun rising over mountain', meaning: 'horizon', phonetic: 'ḥꜣ', category: 'N - Sky, earth, water' },
  'N29': { name: 'Sandy slope', meaning: 'slope', phonetic: 'q', category: 'N - Sky, earth, water' },
  'N31': { name: 'Road with shrubs', meaning: 'road', phonetic: 'wꜣt', category: 'N - Sky, earth, water' },
  'N35': { name: 'Ripple of water', meaning: 'water', phonetic: 'n', category: 'N - Sky, earth, water' },
  'N36': { name: 'Canal', meaning: 'canal', phonetic: 'mr', category: 'N - Sky, earth, water' },
  'N37': { name: 'Pool', meaning: 'pool', phonetic: 'š', category: 'N - Sky, earth, water' },
  'N41': { name: 'Well full of water', meaning: 'well', phonetic: 'ḥm', category: 'N - Sky, earth, water' },
  
  // O - Buildings, parts of buildings
  'O1': { name: 'House', meaning: 'house', phonetic: 'pr', category: 'O - Buildings' },
  'O4': { name: 'Reed shelter', meaning: 'shelter', phonetic: 'h', category: 'O - Buildings' },
  'O6': { name: 'Enclosure', meaning: 'enclosure', phonetic: 'ḥwt', category: 'O - Buildings' },
  'O10': { name: 'Shrine', meaning: 'shrine', phonetic: '', category: 'O - Buildings' },
  'O11': { name: 'Palace', meaning: 'palace', phonetic: '', category: 'O - Buildings' },
  'O19': { name: 'Façade of shrine', meaning: 'shrine', phonetic: '', category: 'O - Buildings' },
  'O20': { name: 'Shrine', meaning: 'shrine', phonetic: '', category: 'O - Buildings' },
  'O21': { name: 'Façade of shrine', meaning: 'shrine', phonetic: '', category: 'O - Buildings' },
  'O28': { name: 'Column', meaning: 'pillar', phonetic: 'iwn', category: 'O - Buildings' },
  'O29': { name: 'Wooden column', meaning: 'pillar', phonetic: 'ꜥꜣ', category: 'O - Buildings' },
  'O31': { name: 'Door', meaning: 'door', phonetic: 'ꜥꜣ', category: 'O - Buildings' },
  'O33': { name: 'Façade of palace', meaning: 'palace', phonetic: '', category: 'O - Buildings' },
  'O34': { name: 'Bolt', meaning: 'bolt', phonetic: 's', category: 'O - Buildings' },
  'O35': { name: 'Door bolt', meaning: 'bolt', phonetic: 'sb', category: 'O - Buildings' },
  'O36': { name: 'Wall', meaning: 'wall', phonetic: 'inb', category: 'O - Buildings' },
  'O38': { name: 'Corner', meaning: 'corner', phonetic: 'qnbt', category: 'O - Buildings' },
  'O39': { name: 'Stone', meaning: 'stone', phonetic: 'inr', category: 'O - Buildings' },
  'O40': { name: 'Stairway', meaning: 'stairway', phonetic: '', category: 'O - Buildings' },
  'O42': { name: 'Fence', meaning: 'fence', phonetic: '', category: 'O - Buildings' },
  'O49': { name: 'Village', meaning: 'town, village', phonetic: 'niwt', category: 'O - Buildings' },
  'O50': { name: 'Threshing floor', meaning: 'floor', phonetic: '', category: 'O - Buildings' },
  
  // P - Ships and parts of ships
  'P1': { name: 'Boat', meaning: 'boat', phonetic: '', category: 'P - Ships' },
  'P3': { name: 'Sacred barque', meaning: 'sacred boat', phonetic: '', category: 'P - Ships' },
  'P5': { name: 'Sail', meaning: 'wind, breath', phonetic: 'nfw', category: 'P - Ships' },
  'P6': { name: 'Mast', meaning: 'stand', phonetic: 'ꜥḥꜥ', category: 'P - Ships' },
  'P8': { name: 'Oar', meaning: 'oar', phonetic: 'ḫrw', category: 'P - Ships' },
  
  // Q - Domestic and funerary furniture
  'Q1': { name: 'Seat', meaning: 'seat', phonetic: 'st', category: 'Q - Furniture' },
  'Q2': { name: 'Carrying chair', meaning: 'carry', phonetic: '', category: 'Q - Furniture' },
  'Q3': { name: 'Stool', meaning: 'stool', phonetic: 'p', category: 'Q - Furniture' },
  'Q4': { name: 'Headrest', meaning: 'rise', phonetic: 'wrs', category: 'Q - Furniture' },
  'Q5': { name: 'Chest', meaning: 'box', phonetic: '', category: 'Q - Furniture' },
  'Q6': { name: 'Coffin', meaning: 'coffin', phonetic: '', category: 'Q - Furniture' },
  'Q7': { name: 'Fire', meaning: 'fire, cook', phonetic: '', category: 'Q - Furniture' },
  
  // R - Temple furniture and sacred emblems
  'R1': { name: 'Flat topped offering table', meaning: 'offering', phonetic: '', category: 'R - Temple furniture' },
  'R3': { name: 'Low table with offerings', meaning: 'offering table', phonetic: '', category: 'R - Temple furniture' },
  'R4': { name: 'Loaf on mat', meaning: 'offering', phonetic: 'ḥtp', category: 'R - Temple furniture' },
  'R5': { name: 'Censer with flame', meaning: 'censer', phonetic: 'kp', category: 'R - Temple furniture' },
  'R7': { name: 'Libation vessel', meaning: 'bowl', phonetic: '', category: 'R - Temple furniture' },
  'R8': { name: 'Standard', meaning: 'god', phonetic: 'nṯr', category: 'R - Temple furniture' },
  'R11': { name: 'Djed pillar', meaning: 'stability', phonetic: 'ḏd', category: 'R - Temple furniture' },
  'R12': { name: 'Standard with feather', meaning: 'west', phonetic: 'imntt', category: 'R - Temple furniture' },
  'R13': { name: 'Falcon standard', meaning: 'Thoth', phonetic: '', category: 'R - Temple furniture' },
  'R14': { name: 'Feather standard', meaning: 'west', phonetic: 'imnt', category: 'R - Temple furniture' },
  'R15': { name: 'Spear standard', meaning: 'east', phonetic: 'iꜣbt', category: 'R - Temple furniture' },
  'R19': { name: 'Was sceptre', meaning: 'power', phonetic: 'wꜣs', category: 'R - Temple furniture' },
  
  // S - Crowns, dress, staves
  'S1': { name: 'White crown', meaning: 'white crown', phonetic: 'ḥḏ', category: 'S - Crowns and dress' },
  'S2': { name: 'White crown', meaning: 'white crown (variant)', phonetic: '', category: 'S - Crowns and dress' },
  'S3': { name: 'Red crown', meaning: 'red crown', phonetic: 'n', category: 'S - Crowns and dress' },
  'S4': { name: 'Red crown', meaning: 'red crown (variant)', phonetic: '', category: 'S - Crowns and dress' },
  'S5': { name: 'Double crown', meaning: 'double crown', phonetic: 'sḫmty', category: 'S - Crowns and dress' },
  'S6': { name: 'Double crown (alt)', meaning: 'the Two Lands', phonetic: '', category: 'S - Crowns and dress' },
  'S12': { name: 'Collar', meaning: 'gold', phonetic: 'nbw', category: 'S - Crowns and dress' },
  'S14': { name: 'Gold sign', meaning: 'gold', phonetic: '', category: 'S - Crowns and dress' },
  'S18': { name: 'Necklace with pendant', meaning: 'ornament', phonetic: 'mnit', category: 'S - Crowns and dress' },
  'S20': { name: 'Seal on necklace', meaning: 'seal', phonetic: 'ḏbꜥ', category: 'S - Crowns and dress' },
  'S22': { name: 'Shoulder knot', meaning: 'knot', phonetic: 'sṯ', category: 'S - Crowns and dress' },
  'S24': { name: 'Girdle knot', meaning: 'tit', phonetic: '', category: 'S - Crowns and dress' },
  'S26': { name: 'Apron', meaning: 'apron', phonetic: '', category: 'S - Crowns and dress' },
  'S29': { name: 'Folded cloth', meaning: 'cloth', phonetic: 's', category: 'S - Crowns and dress' },
  'S34': { name: 'Ankh', meaning: 'life', phonetic: 'ꜥnḫ', category: 'S - Crowns and dress' },
  'S38': { name: 'Crook', meaning: 'ruler', phonetic: 'ḥqꜣ', category: 'S - Crowns and dress' },
  'S39': { name: 'Shepherd crook', meaning: 'rule', phonetic: 'ꜥwt', category: 'S - Crowns and dress' },
  'S40': { name: 'Was sceptre', meaning: 'dominion', phonetic: 'wꜣs', category: 'S - Crowns and dress' },
  'S42': { name: 'Sceptre', meaning: 'sceptre', phonetic: 'ḫrp', category: 'S - Crowns and dress' },
  'S43': { name: 'Walking stick', meaning: 'dignitary', phonetic: 'md', category: 'S - Crowns and dress' },
  'S44': { name: 'Flagellum', meaning: 'flail', phonetic: 'nḫꜣḫꜣ', category: 'S - Crowns and dress' },
  'S45': { name: 'Flagellum (alt)', meaning: 'flail', phonetic: '', category: 'S - Crowns and dress' },
  
  // T - Warfare, hunting, butchery
  'T1': { name: 'Mace with round head', meaning: 'mace', phonetic: '', category: 'T - Warfare' },
  'T3': { name: 'Mace with flat head', meaning: 'mace', phonetic: 'ḥḏ', category: 'T - Warfare' },
  'T7': { name: 'Axe', meaning: 'axe', phonetic: '', category: 'T - Warfare' },
  'T8': { name: 'Dagger', meaning: 'dagger', phonetic: '', category: 'T - Warfare' },
  'T9': { name: 'Bow', meaning: 'bow', phonetic: 'pḏ', category: 'T - Warfare' },
  'T10': { name: 'Composite bow', meaning: 'bow', phonetic: '', category: 'T - Warfare' },
  'T11': { name: 'Arrow', meaning: 'arrow', phonetic: 'sšr', category: 'T - Warfare' },
  'T12': { name: 'Bowstring', meaning: 'bowstring', phonetic: 'rwḏ', category: 'T - Warfare' },
  'T13': { name: 'Joined sticks', meaning: 'border', phonetic: 'rs', category: 'T - Warfare' },
  'T14': { name: 'Throw stick', meaning: 'throw stick, foreigners', phonetic: '', category: 'T - Warfare' },
  'T18': { name: 'Crook with package', meaning: 'shepherd', phonetic: 'sms', category: 'T - Warfare' },
  'T19': { name: 'Harpoon head', meaning: 'harpoon', phonetic: 'qs', category: 'T - Warfare' },
  'T21': { name: 'Harpoon', meaning: 'harpoon', phonetic: 'wꜥ', category: 'T - Warfare' },
  'T22': { name: 'Arrowhead', meaning: 'arrowhead', phonetic: 'sn', category: 'T - Warfare' },
  'T24': { name: 'Fishing net', meaning: 'net', phonetic: '', category: 'T - Warfare' },
  'T25': { name: 'Net on pole', meaning: 'net', phonetic: 'iḥ', category: 'T - Warfare' },
  'T28': { name: 'Butcher block', meaning: 'block', phonetic: 'ḫr', category: 'T - Warfare' },
  'T30': { name: 'Knife', meaning: 'knife', phonetic: '', category: 'T - Warfare' },
  'T31': { name: 'Sharpening stone', meaning: 'sharpen', phonetic: 'sš', category: 'T - Warfare' },
  
  // U - Agriculture, crafts
  'U1': { name: 'Sickle', meaning: 'sickle', phonetic: 'mꜣ', category: 'U - Agriculture' },
  'U6': { name: 'Hoe', meaning: 'hoe', phonetic: 'mr', category: 'U - Agriculture' },
  'U7': { name: 'Hoe (variant)', meaning: 'hoe', phonetic: '', category: 'U - Agriculture' },
  'U13': { name: 'Plough', meaning: 'plough', phonetic: '', category: 'U - Agriculture' },
  'U15': { name: 'Sledge', meaning: 'sledge', phonetic: 'tm', category: 'U - Agriculture' },
  'U17': { name: 'Pick', meaning: 'pick', phonetic: 'grg', category: 'U - Agriculture' },
  'U19': { name: 'Adze', meaning: 'adze', phonetic: 'nw', category: 'U - Agriculture' },
  'U21': { name: 'Adze on block', meaning: 'adze', phonetic: 'stp', category: 'U - Agriculture' },
  'U23': { name: 'Chisel', meaning: 'chisel', phonetic: 'mr', category: 'U - Agriculture' },
  'U26': { name: 'Drill', meaning: 'drill, craftsman', phonetic: '', category: 'U - Agriculture' },
  'U28': { name: 'Fire drill', meaning: 'fire drill', phonetic: 'ḏꜣ', category: 'U - Agriculture' },
  'U30': { name: 'Potters kiln', meaning: 'kiln', phonetic: 'tꜣ', category: 'U - Agriculture' },
  'U32': { name: 'Pestle and mortar', meaning: 'pestle', phonetic: '', category: 'U - Agriculture' },
  'U33': { name: 'Pestle', meaning: 'pestle', phonetic: 'ti', category: 'U - Agriculture' },
  'U36': { name: 'Mallet', meaning: 'mallet', phonetic: 'ḥm', category: 'U - Agriculture' },
  
  // V - Rope, fibre, baskets
  'V1': { name: 'Coil of rope', meaning: 'rope', phonetic: 'šn', category: 'V - Rope' },
  'V4': { name: 'Lasso', meaning: 'lasso', phonetic: 'wꜣ', category: 'V - Rope' },
  'V6': { name: 'Cord', meaning: 'cord', phonetic: 'šs', category: 'V - Rope' },
  'V7': { name: 'Cord wound on stick', meaning: 'cord', phonetic: 'šnw', category: 'V - Rope' },
  'V10': { name: 'Cartouche', meaning: 'cartouche, name', phonetic: '', category: 'V - Rope' },
  'V11': { name: 'Cartouche (ringed)', meaning: 'cartouche', phonetic: '', category: 'V - Rope' },
  'V12': { name: 'String', meaning: 'string', phonetic: '', category: 'V - Rope' },
  'V13': { name: 'Hobble for cattle', meaning: 'hobble', phonetic: 'ṯ', category: 'V - Rope' },
  'V15': { name: 'Hobble with legs', meaning: 'hobble', phonetic: 'iṯ', category: 'V - Rope' },
  'V16': { name: 'Cattle hobble', meaning: 'connect', phonetic: 'sꜣ', category: 'V - Rope' },
  'V20': { name: 'Leather pouch', meaning: 'cover', phonetic: 'mḏ', category: 'V - Rope' },
  'V22': { name: 'Whip', meaning: 'whip', phonetic: 'mḥ', category: 'V - Rope' },
  'V24': { name: 'Cord on stick', meaning: 'cord', phonetic: 'wḏ', category: 'V - Rope' },
  'V25': { name: 'Crossed sticks', meaning: 'cross', phonetic: '', category: 'V - Rope' },
  'V26': { name: 'Block', meaning: 'block', phonetic: 'ꜥḏ', category: 'V - Rope' },
  'V28': { name: 'Wick', meaning: 'wick', phonetic: 'ḥ', category: 'V - Rope' },
  'V29': { name: 'Swab', meaning: 'swab', phonetic: 'sk', category: 'V - Rope' },
  'V30': { name: 'Basket', meaning: 'lord', phonetic: 'nb', category: 'V - Rope' },
  'V31': { name: 'Basket with handle', meaning: 'basket', phonetic: 'k', category: 'V - Rope' },
  
  // W - Vessels of stone and earthenware
  'W1': { name: 'Oil jar', meaning: 'anoint', phonetic: '', category: 'W - Vessels' },
  'W2': { name: 'Oil jar with cover', meaning: 'oil', phonetic: '', category: 'W - Vessels' },
  'W3': { name: 'Alabaster basin', meaning: 'basin', phonetic: 'ḥb', category: 'W - Vessels' },
  'W9': { name: 'Stone jug', meaning: 'jug', phonetic: 'ḫnm', category: 'W - Vessels' },
  'W10': { name: 'Cup', meaning: 'cup', phonetic: 'iꜣb', category: 'W - Vessels' },
  'W11': { name: 'Ring stand', meaning: 'stand', phonetic: 'nst', category: 'W - Vessels' },
  'W14': { name: 'Water jar', meaning: 'water jar', phonetic: 'ḥs', category: 'W - Vessels' },
  'W15': { name: 'Water jar', meaning: 'water pot', phonetic: 'qbḥw', category: 'W - Vessels' },
  'W17': { name: 'Water jars in rack', meaning: 'pure', phonetic: 'ḫnt', category: 'W - Vessels' },
  'W18': { name: 'Water jar in stand', meaning: 'water jar', phonetic: '', category: 'W - Vessels' },
  'W19': { name: 'Milk jug in net', meaning: 'milk', phonetic: 'mi', category: 'W - Vessels' },
  'W22': { name: 'Beer jug', meaning: 'beer', phonetic: '', category: 'W - Vessels' },
  'W24': { name: 'Bowl', meaning: 'bowl', phonetic: 'nw', category: 'W - Vessels' },
  'W25': { name: 'Pot', meaning: 'pot', phonetic: 'ini', category: 'W - Vessels' },
  
  // X - Loaves and cakes
  'X1': { name: 'Bread', meaning: 'bread', phonetic: 't', category: 'X - Loaves' },
  'X2': { name: 'Bread (variant)', meaning: 'bread', phonetic: '', category: 'X - Loaves' },
  'X3': { name: 'Bread on mat', meaning: 'loaf', phonetic: '', category: 'X - Loaves' },
  'X4': { name: 'Cake', meaning: 'give, offer', phonetic: 'ḏi', category: 'X - Loaves' },
  'X5': { name: 'Cake (variant)', meaning: 'offer', phonetic: '', category: 'X - Loaves' },
  'X6': { name: 'Round loaf', meaning: 'bread', phonetic: 'pꜣt', category: 'X - Loaves' },
  'X7': { name: 'Half loaf', meaning: 'half loaf', phonetic: '', category: 'X - Loaves' },
  'X8': { name: 'Conical loaf', meaning: 'give, offering', phonetic: 'ḏi', category: 'X - Loaves' },
  
  // Y - Writings, games, music
  'Y1': { name: 'Papyrus roll', meaning: 'document, writing', phonetic: '', category: 'Y - Writing' },
  'Y2': { name: 'Papyrus roll (tied)', meaning: 'document', phonetic: 'mḏꜣt', category: 'Y - Writing' },
  'Y3': { name: 'Scribe equipment', meaning: 'scribe', phonetic: 'sš', category: 'Y - Writing' },
  'Y4': { name: 'Scribe kit', meaning: 'write', phonetic: '', category: 'Y - Writing' },
  'Y5': { name: 'Senet board', meaning: 'game, play', phonetic: 'mn', category: 'Y - Writing' },
  'Y6': { name: 'Game piece', meaning: 'game', phonetic: '', category: 'Y - Writing' },
  'Y8': { name: 'Sistrum', meaning: 'music', phonetic: 'sššt', category: 'Y - Writing' },
  
  // Z - Strokes, geometrical figures
  'Z1': { name: 'Stroke', meaning: 'one', phonetic: '', category: 'Z - Strokes' },
  'Z2': { name: 'Plural strokes', meaning: 'plural', phonetic: '', category: 'Z - Strokes' },
  'Z3': { name: 'Three strokes', meaning: 'plural', phonetic: '', category: 'Z - Strokes' },
  'Z4': { name: 'Dual strokes', meaning: 'dual', phonetic: 'y', category: 'Z - Strokes' },
  'Z5': { name: 'Diagonal line', meaning: 'damage', phonetic: '', category: 'Z - Strokes' },
  'Z6': { name: 'Substitute ideogram', meaning: 'substitute', phonetic: '', category: 'Z - Strokes' },
  'Z7': { name: 'Coil', meaning: 'coil', phonetic: 'w', category: 'Z - Strokes' },
  'Z9': { name: 'Two diagonal lines', meaning: 'break', phonetic: '', category: 'Z - Strokes' },
  'Z11': { name: 'Two planks', meaning: 'cross', phonetic: 'imi', category: 'Z - Strokes' },
  
  // Aa - Unclassified
  'Aa1': { name: 'Placenta', meaning: 'placenta', phonetic: 'ḫ', category: 'Aa - Unclassified' },
  'Aa2': { name: 'Pustule', meaning: 'pustule', phonetic: '', category: 'Aa - Unclassified' },
  'Aa5': { name: 'Part of steering oar', meaning: 'steering', phonetic: 'ḥpt', category: 'Aa - Unclassified' },
  'Aa8': { name: 'Pool with lotus', meaning: 'pool', phonetic: '', category: 'Aa - Unclassified' },
  'Aa11': { name: 'Platform', meaning: 'platform', phonetic: 'mꜣꜥ', category: 'Aa - Unclassified' },
  'Aa13': { name: 'Final', meaning: 'back', phonetic: '', category: 'Aa - Unclassified' },
  'Aa15': { name: 'Divinity back', meaning: 'final', phonetic: '', category: 'Aa - Unclassified' },
  'Aa17': { name: 'Lid', meaning: 'lid', phonetic: '', category: 'Aa - Unclassified' },
  'Aa20': { name: 'Shrine', meaning: 'shrine', phonetic: '', category: 'Aa - Unclassified' },
  'Aa21': { name: 'Shrine', meaning: 'shrine', phonetic: 'wḏꜣ', category: 'Aa - Unclassified' },
  'Aa26': { name: 'Unknown', meaning: 'unknown', phonetic: '', category: 'Aa - Unclassified' },
  'Aa27': { name: 'Unknown', meaning: 'unknown', phonetic: 'nḏ', category: 'Aa - Unclassified' },
  'Aa28': { name: 'Unknown', meaning: 'unknown', phonetic: 'qd', category: 'Aa - Unclassified' },
};

// Get all unique categories
export const categories = [...new Set(Object.values(gardinerSigns).map(s => s.category))].sort();

// Get signs by category
export function getSignsByCategory(category) {
  return Object.entries(gardinerSigns)
    .filter(([_, sign]) => sign.category === category)
    .map(([code, sign]) => ({ code, ...sign }));
}

// Search signs
export function searchSigns(query) {
  const lowerQuery = query.toLowerCase();
  return Object.entries(gardinerSigns)
    .filter(([code, sign]) => 
      code.toLowerCase().includes(lowerQuery) ||
      sign.name.toLowerCase().includes(lowerQuery) ||
      sign.meaning.toLowerCase().includes(lowerQuery) ||
      sign.phonetic.toLowerCase().includes(lowerQuery)
    )
    .map(([code, sign]) => ({ code, ...sign }));
}

export default gardinerSigns;
