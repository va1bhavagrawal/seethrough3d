scales = {
    "bear": 0.53,       # Unchanged
    "bicycle": 0.4,      # Unchanged
    "bugatti": 1.0,      # Unchanged
    "bulldozer": 1.78,   # Unchanged
    "bus": 2.67,         # Unchanged
    "cat": 0.11,         # Unchanged
    "chair": 0.18,       # Unchanged
    "coupe": 1.0,        # Unchanged
    "cow": 0.56,         # Unchanged
    "crow": 0.09,        # CHANGED: Reduced from 0.11
    "deer": 0.44,        # Unchanged
    "dog": 0.22,         # Unchanged
    "elephant": 1.22,    # Unchanged
    "ferrari": 1.05,     # CHANGED: Increased from 1.0
    "flamingo": 0.10,    # Unchanged
    "fox": 0.22,         # Unchanged
    "giraffe": 0.90,     # CHANGED: Reduced from 1.00
    "goat": 0.33,        # Unchanged
    "helicopter": 2.26,  # Unchanged
    "hen": 0.09,         # Unchanged
    "horse": 0.53,       # Unchanged
    "jeep": 0.96,        # Unchanged
    "kangaroo": 0.38,    # CHANGED: Increased from 0.33
    "lamborghini": 1.0,  # Unchanged
    "lion": 0.56,        # Unchanged
    "mclaren": 1.0,      # Unchanged
    "motorbike": 0.44,   # Unchanged
    "office chair": 0.20,# Unchanged
    "pickup truck": 1.22,# Unchanged
    "pigeon": 0.067,     # Unchanged
    "pig": 0.33,         # Unchanged
    "rabbit": 0.11,      # Unchanged
    "scooter": 0.4,      # Unchanged
    "sedan": 1.0,        # Unchanged (Reference)
    "sheep": 0.29,       # Unchanged
    "shoe": 0.04,        # Unchanged
    "sparrow": 0.033,    # Unchanged
    "suv": 1.07,         # Unchanged
    "table": 0.4,        # Unchanged
    "teddy": 0.05,       # CHANGED: Reduced from 0.11
    "tiger": 0.67,       # Unchanged
    "tractor": 0.80,     # Unchanged
    "van": 1.11,         # Unchanged
    "vw beetle": 1.0,    # Unchanged
    "wolf": 0.33,        # Unchanged
    "man": 0.38,         # Unchanged
    "zebra": 0.56        # Unchanged
}

tiny_assets = [
    "sparrow",      # 0.033
    "shoe",         # 0.04
    "teddy",        # 0.05 (CHANGED)
    "pigeon",       # 0.067
    "hen",          # 0.09
    "crow",         # 0.09 (CHANGED)
    "flamingo",     # 0.10 (CHANGED - Moved from small)
    "rabbit",       # 0.11
    "cat",          # 0.11
]


small_assets = [
    "chair",        # 0.18
    "office chair", # 0.20
    "dog",          # 0.22
    "fox",          # 0.22
    "sheep",        # 0.29
    "goat",         # 0.33
    "pig",          # 0.33
    "wolf",         # 0.33
    "man",          # 0.38 (CHANGED - Added to group)
    "kangaroo",     # 0.38 (CHANGED)
]


medium_assets = [
    "table",        # 0.4 (CHANGED - Moved from small)
    "bicycle",      # 0.4
    "scooter",      # 0.4
    "deer",         # 0.44
    "motorbike",    # 0.44
    "bear",         # 0.53
    "horse",        # 0.53
    "cow",          # 0.56
    "lion",         # 0.56
    "zebra",        # 0.56
    "tiger",        # 0.67
    "tractor",      # 0.80
    "giraffe",      # 0.90 (CHANGED)
    "jeep",         # 0.96
    "bugatti",      # 1.0
    "coupe",        # 1.0
    "lamborghini",  # 1.0
    "mclaren",      # 1.0
    "sedan",        # 1.0
    "vw beetle",    # 1.0
    "ferrari",      # 1.05 (CHANGED)
    "suv",          # 1.07
    "van",          # 1.11
    "elephant",     # 1.22
    "pickup truck", # 1.22
    "bulldozer",    # 1.78
    "helicopter",   # 2.26
    "bus",          # 2.67
]

tiny_prompts = [
    "a photo of PLACEHOLDER in a cozy birdhouse nestled in a green tree",
    "a photo of PLACEHOLDER on a sandy beach near the water's edge with small shells",
    "a photo of PLACEHOLDER amongst colorful wildflowers in a sunny meadow",
    "a photo of PLACEHOLDER on a moss-covered log in a quiet forest",
    "a photo of PLACEHOLDER near a small pond with lily pads floating",
    "a photo of PLACEHOLDER on a window sill overlooking a rainy city street",
    "a photo of PLACEHOLDER in a child's bedroom surrounded by other toys",
    "a photo of PLACEHOLDER on a park bench with fallen leaves around",
    "a photo of PLACEHOLDER by a small stream with smooth pebbles",
    "a photo of PLACEHOLDER in a field of tall grass swaying gently",
    "a photo of PLACEHOLDER on a wooden fence post in the countryside",
    "a photo of PLACEHOLDER amongst blossoming spring flowers in a garden",
    "a photo of PLACEHOLDER on a stack of old books in a library",
    "a photo of PLACEHOLDER near a bird feeder in a winter garden",
    "a photo of PLACEHOLDER on a picnic blanket in a sunny park",
    "a photo of PLACEHOLDER on a kitchen counter near ripe fruit",
    "a photo of PLACEHOLDER amongst autumn leaves on a forest floor",
    "a photo of PLACEHOLDER on a rocky outcrop with a distant view",
    "a photo of PLACEHOLDER near a puddle reflecting the sky",
    "a photo of PLACEHOLDER in a patch of soft green moss",
    "a photo of PLACEHOLDER on a weathered stone wall",
    "a photo of PLACEHOLDER near a patch of blooming daisies",
    "a photo of PLACEHOLDER on a sandy path through a garden",
    "a photo of PLACEHOLDER near a watering can in a greenhouse",
    "a photo of PLACEHOLDER amongst fallen pine needles in a forest",
    "a photo of PLACEHOLDER on a small bridge over a gentle stream",
    "a photo of PLACEHOLDER near a patch of colorful mushrooms"
]

small_prompts = [
    "a photo of PLACEHOLDER in a sun-drenched greenhouse surrounded by various plants",
    "a photo of PLACEHOLDER in a bustling city park with people walking by",
    "a photo of PLACEHOLDER in a cozy library with tall bookshelves and soft lighting",
    "a photo of PLACEHOLDER on a sandy dune near the ocean with gentle waves",
    "a photo of PLACEHOLDER amongst tall reeds in a marshland area",
    "a photo of PLACEHOLDER in a quiet forest clearing with sunlight filtering through trees",
    "a photo of PLACEHOLDER on a grassy hill overlooking a small town",
    "a photo of PLACEHOLDER near a flowing waterfall with mist in the air",
    "a photo of PLACEHOLDER in a vibrant flower market with colorful blooms all around",
    "a photo of PLACEHOLDER on a wooden dock extending into a still lake",
    "a photo of PLACEHOLDER amongst rows of crops in a rural farmland",
    "a photo of PLACEHOLDER in a historic town square with old buildings",
    "a photo of PLACEHOLDER on a rocky beach with crashing waves in the distance",
    "a photo of PLACEHOLDER amongst tall bamboo stalks in a serene grove",
    "a photo of PLACEHOLDER in a snowy field with tracks visible in the snow",
    "a photo of PLACEHOLDER on a paved walkway in a botanical garden",
    "a photo of PLACEHOLDER near a campfire in a forest at night",
    "a photo of PLACEHOLDER amongst colorful autumn foliage in a park",
    "a photo of PLACEHOLDER on a stone path winding through a garden",
    "a photo of PLACEHOLDER in a misty meadow with dew-covered grass",
    "a photo of PLACEHOLDER on a wooden bridge crossing a small river",
    "a photo of PLACEHOLDER amongst blooming lavender fields under a sunny sky",
    "a photo of PLACEHOLDER in a quiet suburban backyard with green grass",
    "a photo of PLACEHOLDER on a rocky hillside with sparse vegetation",
    "a photo of PLACEHOLDER near a clear mountain stream with smooth stones",
    "a photo of PLACEHOLDER amongst fallen leaves in a shaded woodland",
    "a photo of PLACEHOLDER on a grassy bank beside a calm canal",
    "a photo of PLACEHOLDER in a vineyard with rows of grapevines",
    "a photo of PLACEHOLDER near a traditional wooden farmhouse"
]

medium_prompts = [
    "a photo of PLACEHOLDER in a vast open plain with a dramatic sunset on the horizon",
    "a photo of PLACEHOLDER on a winding mountain road with scenic views of valleys",
    "a photo of PLACEHOLDER in a bustling harbor with various boats and ships",
    "a photo of PLACEHOLDER in a dense pine forest with tall trees reaching the sky",
    "a photo of PLACEHOLDER on a sandy beach with palm trees swaying in the breeze",
    "a photo of PLACEHOLDER amongst rolling hills in a green countryside landscape",
    "a photo of PLACEHOLDER in a vibrant city square with historic architecture",
    "a photo of PLACEHOLDER in a train yard with multiple railway tracks", 
    "a photo of PLACEHOLDER amongst tall redwood trees in an ancient forest",
    "a photo of PLACEHOLDER in a sprawling parking lot outside a shopping mall", 
    "a photo of PLACEHOLDER on a coastal highway with ocean views and cliffs", 
    "a photo of PLACEHOLDER amongst golden wheat fields under a clear summer sky",
    "a photo of PLACEHOLDER in a rocky canyon with sparse desert vegetation and blue sky above",
    "a photo of PLACEHOLDER on a grassy plateau overlooking a vast landscape",
    "a photo of PLACEHOLDER in a snowy mountain range with visible ski slopes",
    "a photo of PLACEHOLDER on a paved highway stretching across an open landscape",
    "a photo of PLACEHOLDER amongst lush vegetation in a tropical rainforest",
    "a photo of PLACEHOLDER in a historic European city with ornate buildings",
    "a photo of PLACEHOLDER in front of the Eiffel Tower at sunset",
    "a photo of PLACEHOLDER amongst tall sunflowers in a field under a bright sun",
    "a photo of PLACEHOLDER in a deep valley with steep forested sides",
    "a photo of PLACEHOLDER on a rocky coastline with crashing waves and sea spray",
    "a photo of PLACEHOLDER amongst vineyards on rolling hills under a sunny sky",
    "a photo of PLACEHOLDER in a wide open desert with distant mesas and clear air",
    "a photo of PLACEHOLDER amongst autumn-colored trees along a winding river",
    "a photo of PLACEHOLDER in a bustling marketplace with various stalls and people",
    "a photo of PLACEHOLDER on a racing circuit with banked turns and grandstands", 
    "a photo of PLACEHOLDER amongst tall grasses in a savanna landscape", 
]

groups = {
    "tiny": tiny_assets,
    "small": small_assets,
    "medium": medium_assets,
}

groups_prompts = {
    "tiny": tiny_prompts,  
    "small": small_prompts, 
    "medium": medium_prompts, 
}