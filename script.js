let model;
let classLabels = [];
const upload = document.getElementById('upload');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

async function loadClassLabels() {
  try {
    // Use a reliable GitHub raw content URL for ImageNet labels
    const response = await fetch('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    classLabels = await response.json();
    console.log('Class labels loaded successfully');
  } catch (error) {
    console.error('Error loading class labels:', error);
    // Fallback: use a hardcoded subset of common ImageNet labels
    classLabels = [
      'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
      'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch',
      'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie',
      'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl',
      'fire salamander', 'smooth newt', 'newt', 'spotted salamander', 'axolotl',
      'bullfrog', 'tree frog', 'tailed frog', 'loggerhead turtle', 'leatherback turtle',
      'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana',
      'American chameleon', 'whiptail lizard', 'agama', 'frilled lizard', 'alligator lizard',
      'Gila monster', 'green lizard', 'African chameleon', 'Komodo dragon', 'African crocodile',
      'American alligator', 'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake',
      'green snake', 'king snake', 'garter snake', 'water snake', 'vine snake', 'night snake',
      'boa constrictor', 'rock python', 'Indian cobra', 'green mamba', 'sea snake',
      'horned viper', 'diamondback rattlesnake', 'sidewinder', 'trilobite', 'harvestman',
      'scorpion', 'black and gold garden spider', 'barn spider', 'garden spider', 'black widow',
      'tarantula', 'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse',
      'prairie chicken', 'peacock', 'quail', 'partridge', 'African grey parrot', 'macaw',
      'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird',
      'jacamar', 'toucan', 'duck', 'red-breasted merganser', 'goose', 'black swan', 'tusker',
      'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral',
      'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus',
      'Dungeness crab', 'rock crab', 'fiddler crab', 'king crab', 'American lobster', 'spiny lobster',
      'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo',
      'little blue heron', 'American egret', 'bittern', 'crane', 'limpkin', 'American coot',
      'bustard', 'ruddy turnstone', 'red-backed sandpiper', 'redshank', 'dowitcher', 'oystercatcher',
      'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong', 'sea lion',
      'Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu', 'Blenheim spaniel',
      'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'basset hound', 'beagle',
      'bloodhound', 'bluetick coonhound', 'black-and-tan coonhound', 'Walker hound', 'English foxhound',
      'redbone coonhound', 'borzoi', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound',
      'Norwegian elkhound', 'otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier',
      'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier',
      'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier',
      'Lakeland terrier', 'Sealyham terrier', 'Airedale terrier', 'cairn terrier', 'Australian terrier',
      'Dandie Dinmont terrier', 'Boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer',
      'Scotch terrier', 'Tibetan terrier', 'silky terrier', 'soft-coated wheaten terrier', 'West Highland white terrier',
      'Lhasa apso', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever',
      'Chesapeake Bay retriever', 'German short-haired pointer', 'Hungarian pointer', 'English setter',
      'Irish setter', 'Gordon setter', 'Brittany spaniel', 'clumber spaniel', 'English springer spaniel',
      'Welsh springer spaniel', 'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke',
      'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog', 'Shetland sheepdog',
      'collie', 'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman pinscher',
      'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher',
      'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog',
      'malamute', 'Siberian husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees',
      'Samoyed', 'Pomeranian', 'chow chow', 'keeshond', 'Brabancon griffon', 'Pembroke Welsh corgi', 'Cardigan Welsh corgi',
      'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf', 'white wolf', 'red wolf',
      'coyote', 'dingo', 'dhole', 'African hunting dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox', 'grey fox', 'tabby cat',
      'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian cat', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar',
      'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear', 'ice bear', 'sloth bear', 'mongoose', 'meerkat',
      'tiger beetle', 'ladybug', 'ground beetle', 'long-horned beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle',
      'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking stick', 'cockroach', 'mantis', 'cicada', 'leafhopper',
      'lacewing', 'dragonfly', 'damselfly', 'admiral butterfly', 'ringlet butterfly', 'monarch butterfly', 'cabbage butterfly',
      'sulphur butterfly', 'lycaenid butterfly', 'starfish', 'sea urchin', 'sea cucumber', 'wood rabbit', 'hare', 'Angora rabbit',
      'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'sorrel horse', 'zebra', 'hog', 'wild boar',
      'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn sheep', 'ibex', 'hartebeest', 'impala',
      'gazelle', 'Arabian camel', 'llama', 'weasel', 'mink', 'polecat', 'black-footed ferret', 'otter', 'skunk', 'badger',
      'armadillo', 'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas monkey',
      'baboon', 'macaque', 'langur', 'colobus monkey', 'proboscis monkey', 'marmoset', 'capuchin monkey', 'howler monkey',
      'titi monkey', 'spider monkey', 'squirrel monkey', 'Madagascar cat', 'indri', 'Indian elephant', 'African elephant',
      'lesser panda', 'giant panda', 'barracouta', 'eel', 'coho salmon', 'rock beauty fish', 'anemone fish', 'sturgeon',
      'gar fish', 'lionfish', 'pufferfish', 'abacus', 'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier',
      'airliner', 'airship', 'altar', 'ambulance', 'amphibian vehicle', 'analog clock', 'apiary', 'apron', 'ashcan', 'assault rifle',
      'backpack', 'bakery', 'balance beam', 'balloon', 'ballpoint pen', 'Band Aid', 'banjo', 'bannister', 'barbell', 'barber chair',
      'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap',
      'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker', 'military cap', 'fire engine', 'beer bottle', 'beer glass',
      'bell cote', 'bib', 'tandem bicycle', 'bikini', 'binder', 'binoculars', 'birdhouse', 'boathouse', 'bobsled', 'bolo tie',
      'bonnet', 'bookcase', 'bookstore', 'bottle cap', 'bow', 'bow tie', 'brass', 'bra', 'breakwater', 'breastplate', 'broom',
      'bucket', 'buckle', 'bulletproof vest', 'high-speed train', 'butcher shop', 'taxi', 'cauldron', 'candle', 'cannon', 'canoe',
      'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit', 'carton', 'car wheel', 'automated teller machine', 'cassette',
      'cassette player', 'castle', 'catamaran', 'CD player', 'cello', 'mobile phone', 'chain', 'chain-link fence', 'chain mail',
      'chain saw', 'chest', 'chiffonier', 'chime', 'china cabinet', 'Christmas stocking', 'church', 'cinema', 'cleaver', 'cliff dwelling',
      'cloak', 'clog', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil', 'combination lock', 'computer keyboard', 'confectionery',
      'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle', 'crane (machine)', 'crash helmet',
      'crate', 'infant bed', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'rotary dial telephone',
      'diaper', 'digital clock', 'digital watch', 'dining table', 'dishrag', 'dishwasher', 'disc brake', 'dock', 'dog sled', 'dome',
      'doormat', 'drilling platform', 'drum', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar', 'electric locomotive',
      'entertainment center', 'envelope', 'espresso machine', 'face powder', 'feather boa', 'filing cabinet', 'fireboat', 'fire engine',
      'fire screen', 'flagpole', 'flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster bed',
      'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck', 'gas mask', 'gas pump', 'goblet', 'go-kart', 'golf ball',
      'golf cart', 'gondola', 'gong', 'gown', 'grand piano', 'greenhouse', 'grille', 'grocery store', 'guillotine', 'hair slide',
      'hair spray', 'half track', 'hammer', 'hamper', 'hair dryer', 'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica',
      'harp', 'harvester', 'hatchet', 'holster', 'home theater', 'honeycomb', 'hook', 'hoop skirt', 'horizontal bar', 'horse cart',
      'hourglass', 'iPod', 'iron', 'jack-o\'-lantern', 'jean', 'jeep', 'T-shirt', 'jigsaw puzzle', 'pulled rickshaw', 'joystick',
      'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade', 'laptop', 'lawn mower', 'lens cap', 'letter opener', 'library',
      'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick', 'slip-on shoe', 'lotion', 'loudspeaker', 'loupe', 'sawmill',
      'magnetic compass', 'mailbag', 'mailbox', 'tights', 'tank suit', 'manhole cover', 'maraca', 'marimba', 'mask', 'matchstick',
      'maypole', 'maze', 'measuring cup', 'medicine chest', 'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can',
      'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home', 'Model T', 'modem', 'monastery', 'monitor',
      'moped', 'mortar', 'square academic cap', 'mosque', 'mosquito net', 'motor scooter', 'mountain bike', 'tent', 'computer mouse',
      'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook computer', 'obelisk', 'oboe', 'ocarina',
      'odometer', 'oil filter', 'organ', 'oscilloscope', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle', 'paddle wheel', 'padlock',
      'paintbrush', 'pajama', 'palace', 'pan flute', 'paper towel', 'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car',
      'patio', 'payphone', 'pedestal', 'pencil case', 'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'pick', 'pickelhaube',
      'picket fence', 'pickup truck', 'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship', 'pitcher',
      'plane', 'planetarium', 'plastic bag', 'plate rack', 'plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho', 'pool table',
      'soda bottle', 'pot', 'potter\'s wheel', 'power drill', 'prayer rug', 'printer', 'prison', 'projectile', 'projector', 'hockey puck',
      'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'recreational vehicle',
      'reel', 'reflex camera', 'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle', 'rocking chair', 'rotisserie', 'eraser',
      'rugby ball', 'ruler', 'running shoe', 'safe', 'safety pin', 'salt shaker', 'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale',
      'school bus', 'schooner', 'scoreboard', 'screen', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe shop', 'shoji',
      'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule', 'sliding door',
      'slot machine', 'snorkel', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock', 'solar dish', 'sombrero', 'soup bowl', 'space bar',
      'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web', 'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive',
      'through arch bridge', 'steel drum', 'stethoscope', 'scarf', 'stone wall', 'stopwatch', 'stove', 'strainer', 'tram', 'stretcher', 'couch',
      'stupa', 'submarine', 'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension bridge', 'swab', 'sweatshirt', 'swimming trunks',
      'swing', 'switch', 'syringe', 'table lamp', 'tank', 'tape player', 'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof',
      'front curtain', 'thimble', 'threshing machine', 'throne', 'tile roof', 'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole',
      'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch',
      'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase',
      'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe',
      'military aircraft', 'sink', 'washing machine', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen',
      'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence', 'shipwreck', 'yurt', 'website',
      'comic book', 'crossword puzzle', 'traffic light', 'dust jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream',
      'ice lolly', 'French loaf', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'mashed potato', 'cabbage', 'broccoli', 'cauliflower', 'zucchini',
      'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith apple',
      'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce',
      'dough', 'meat loaf', 'pizza', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff', 'coral reef', 'geyser',
      'lakeside', 'promontory', 'sandbar', 'seashore', 'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver', 'rapeseed', 'daisy',
      'yellow lady\'s slipper', 'corn', 'acorn', 'rose hip', 'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn mushroom',
      'earthstar fungus', 'hen-of-the-woods', 'bolete', 'ear', 'toilet tissue'
    ];
  }
}

async function loadModel() {
  result.textContent = 'Loading MobileNet model...';
  
  try {
    // Load class labels first
    await loadClassLabels();
    
    // Use MobileNetV1 which has a reliable URL
    model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    result.textContent = 'Model loaded. Upload an image to classify.';
  } catch (error) {
    console.error('Error loading model:', error);
    result.textContent = 'Error loading model. Please refresh the page.';
  }
}

upload.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  reader.onload = async function () {
    preview.src = reader.result;
    result.textContent = 'Analyzing...';

    preview.onload = async () => {
      if (!model) {
        result.textContent = 'Model not loaded yet. Please wait.';
        return;
      }

      if (classLabels.length === 0) {
        result.textContent = 'Class labels not loaded yet. Please wait.';
        return;
      }

      try {
        // Preprocess image for MobileNetV1
        const tensor = tf.browser.fromPixels(preview)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .expandDims(0)
          .div(255); // Normalize to [0, 1]

        // Get predictions
        const predictions = model.predict(tensor);
        const data = await predictions.data();

        // Get top 5 predictions with class names
        const top5 = Array.from(data)
          .map((probability, index) => ({ 
            probability, 
            className: classLabels[index] || `Class ${index}`,
            classIndex: index 
          }))
          .sort((a, b) => b.probability - a.probability)
          .slice(0, 5);

        // Display results with class names
        result.innerHTML = '<strong>Top 5 Predictions:</strong><br>' + 
          top5.map((p, i) => `
            <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
              <div style="font-weight: bold; margin-bottom: 5px;">
                #${i + 1}: ${p.className}
              </div>
              <div style="color: #666; font-size: 14px; margin-bottom: 5px;">
                Confidence: ${(p.probability * 100).toFixed(2)}% (Class ${p.classIndex})
              </div>
              <div class="bar" style="width:${Math.min(p.probability * 100, 100)}%;"></div>
            </div>`).join('');

        // Clean up
        tensor.dispose();
        predictions.dispose();

      } catch (error) {
        console.error('Error during prediction:', error);
        result.textContent = 'Error analyzing image.';
      }
    };
  };
  reader.readAsDataURL(file);
});

loadModel();
