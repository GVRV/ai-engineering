# Chunking Exercise

Q: Are there 2 main stages of photosynthesis?
Whole chunks: [(0.5145775675773621, 'WHOLE')]
Fixed Window chunks:[(0.8370662331581116, 'FIXED_WINDOW_33'), (0.7011384963989258, 'FIXED_WINDOW_32'), (0.6704960465431213, 'FIXED_WINDOW_47')]
Section chunks: [(0.679176926612854, 'SECTION_4'), (0.6084101796150208, 'SECTION_2'), (0.6048596501350403, 'SECTION_5')]


Q: Is chlorophyll responsible for a plant's green colour?
Whole chunks: [(0.42783886194229126, 'WHOLE')]
Fixed Window chunks:[(0.7867172360420227, 'FIXED_WINDOW_29'), (0.6755842566490173, 'FIXED_WINDOW_30'), (0.6364982724189758, 'FIXED_WINDOW_37')]
Section chunks: [(0.766249418258667, 'SECTION_3'), (0.4792312979698181, 'SECTION_4'), (0.4402560591697693, 'SECTION_1')]


Q: Does most life on Earth depend on photosynthesis?
Whole chunks: [(0.6399862170219421, 'WHOLE')]
Fixed Window chunks:[(0.6357166767120361, 'FIXED_WINDOW_17'), (0.6245216131210327, 'FIXED_WINDOW_9'), (0.602301836013794, 'FIXED_WINDOW_46')]
Section chunks: [(0.6510334014892578, 'SECTION_1'), (0.5002252459526062, 'SECTION_2'), (0.4868522584438324, 'SECTION_4')]


Q: Plants make food using chrolophyll and photosynthesis only in the leaves, and only in the day. What should I correct, for class 7?
Whole chunks: [(0.5569363832473755, 'WHOLE')]
Fixed Window chunks:[(0.6420378684997559, 'FIXED_WINDOW_1'), (0.5851877927780151, 'FIXED_WINDOW_0'), (0.5694276690483093, 'FIXED_WINDOW_58')]
Section chunks: [(0.5683802366256714, 'SECTION_1'), (0.5639815330505371, 'SECTION_3'), (0.543097198009491, 'SECTION_6')]


# Completion Requests

## Fixed Window
Q: Is chlorophyll responsible for a plant's green colour?
Reply: {'file': 'FIXED_WINDOW_29', 'score': 0.7867172360420227, 'reply': 'Yes. The source says that chlorophyll is “for giving the plant its green color.”'}
Read Grade: No (we can't see from the chunk what exactly is giving the plant its green color)
System Grade: Yes
Source:
 for giving the plant its green color. During photosynthesis, chlorophyll absorb

## Section
Q: Is chlorophyll responsible for a plant's green colour?
Reply: {'file': 'SECTION_3', 'score': 0.766249418258667, 'reply': 'Yes. The source says that chlorophyll is the pigment that gives a plant its green colour.'}
Read Grade: Yes
System Grade: Yes
Source:
Chlorophyll

Inside the plant cell are small organelles called chloroplasts, which store the energy of sunlight. Within the thylakoid membranes of the chloroplast is a light-absorbing pigment called chlorophyll, which is responsible for giving the plant its green color. During photosynthesis, chlorophyll absorbs energy from blue- and red-light waves, and reflects green-light waves, making the plant appear green.


# Multiple Question Eval

question: Are the chemicals involved in photosynthesis carbon dioxide and oxygen?
expected section id: 2
top section id: 2
top window id: 18
hit: yes

question: Is photosynthesis less effective during winter months?
question: Are the chemicals involved in photosynthesis carbon dioxide and oxygen?
expected section id, or none: 2
top section id: SECTION_2
top window id: FIXED_WINDOW_18
hit: True

question: Is photosynthesis less effective during winter months?
expected section id, or none: None
top section id: SECTION_2
top window id: FIXED_WINDOW_55
hit: False

question: Is chlorophyll responsible for a plant's green colour?
expected section id, or none: 3
top section id: SECTION_3
top window id: FIXED_WINDOW_29
hit: True

question: Are fruits where plant stores the energy produced by photosynthesis?
expected section id, or none: 6
top section id: SECTION_6
top window id: FIXED_WINDOW_1
hit: True

question: Does most life on Earth depend on photosynthesis?
expected section id, or none: 1
top section id: SECTION_1
top window id: FIXED_WINDOW_17
hit: True

question: Are school grammar classes voluntary for students to attend?
expected section id, or none: None
top section id: SECTION_1
top window id: FIXED_WINDOW_47
hit: False

question: Did Gandhi's plant based diet give him the energy to fight for independence?
expected section id, or none: None
top section id: SECTION_1
top window id: FIXED_WINDOW_6
hit: False

question: Plants are green because that colour helps with plant reproduction?
expected section id, or none: None
top section id: SECTION_3
top window id: FIXED_WINDOW_31
hit: False

question: The water cycle is essential for all plants to survive. True or False?
expected section id, or none: 1
top section id: SECTION_1
top window id: FIXED_WINDOW_57
hit: True

question: Who was the first human to land on the moon?
expected section id, or none: None
top section id: SECTION_0
top window id: FIXED_WINDOW_7
hit: False