QUESTION = """A student says plants make food only in the leaves, and only in the day. What should I correct, for class 7?"""

from client import complete

print("Bare:")
print(complete("", QUESTION))

print("Role + Constraints:")
system_prompt = """"You are a teacher’s assistant for ICSE class 7. Use simple English. Do not invent a page number. If you are not sure, say so. Max 120 words."""
print(complete(system_prompt, QUESTION))

print("Page in the window")
user_prompt = f"""
Most life on Earth depends on photosynthesis.The process is carried out by plants, algae, and some types of bacteria, which capture energy from sunlight to produce oxygen (O2) and chemical energy stored in glucose (a sugar). Herbivores then obtain this energy by eating plants, and carnivores obtain it by eating herbivores.

The process

During photosynthesis, plants take in carbon dioxide (CO2) and water (H2O) from the air and soil. Within the plant cell, the water is oxidized, meaning it loses electrons, while the carbon dioxide is reduced, meaning it gains electrons. This transforms the water into oxygen and the carbon dioxide into glucose. The plant then releases the oxygen back into the air, and stores energy within the glucose molecules.

Chlorophyll

Inside the plant cell are small organelles called chloroplasts, which store the energy of sunlight. Within the thylakoid membranes of the chloroplast is a light-absorbing pigment called chlorophyll, which is responsible for giving the plant its green color. During photosynthesis, chlorophyll absorbs energy from blue- and red-light waves, and reflects green-light waves, making the plant appear green.

Light-dependent Reactions vs. Light-independent Reactions

While there are many steps behind the process of photosynthesis, it can be broken down into two major stages: light-dependent reactions and light-independent reactions. The light-dependent reaction takes place within the thylakoid membrane and requires a steady stream of sunlight, hence the name light-dependent reaction. The chlorophyll absorbs energy from the light waves, which is converted into chemical energy in the form of the molecules ATP and NADPH. The light-independent stage, also known as the Calvin cycle, takes place in the stroma, the space between the thylakoid membranes and the chloroplast membranes, and does not require light, hence the name light-independent reaction. During this stage, energy from the ATP and NADPH molecules is used to assemble carbohydrate molecules, like glucose, from carbon dioxide.

C3 and C4 Photosynthesis

Not all forms of photosynthesis are created equal, however. There are different types of photosynthesis, including C3 photosynthesis and C4 photosynthesis. C3 photosynthesis is used by the majority of plants. It involves producing a three-carbon compound called 3-phosphoglyceric acid during the Calvin Cycle, which goes on to become glucose. C4 photosynthesis, on the other hand, produces a four-carbon intermediate compound, which splits into carbon dioxide and a three-carbon compound during the Calvin Cycle. A benefit of C4 photosynthesis is that by producing higher levels of carbon, it allows plants to thrive in environments without much light or water.

{QUESTION}

Answer from this paragraph.
"""
print(complete(system_prompt, user_prompt))