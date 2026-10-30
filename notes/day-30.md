The system is three calls in a fixed order. Embed the question, take the top paragraph, ask the answer model, ask the judge.

The judge sees the paste and the reply. It does not see the gold span.

A YES from the judge is not a correct answer. It is “supported by this paste.” Rainforest can still come back YES. That row stays a known failure.

# Notes

Guard confirmed on Yuan: Yes
Command: uv run python loop.py <question>
Two demo outputs, labeled:
- `loop.py "What day was the Super Bowl played on?"`
    Model Reply: NOT_IN_SOURCE
    Grounded Reply Judgement: NO; paragraph contains the answer.
    Paste Context: The game's media day, which was typically held on the Tuesday afternoon prior to the game, was moved to the Monday evening and re-branded as Super Bowl Opening Night. The event was held on February 1, 2016 at SAP Center in San Jose. Alongside the traditional media availabilities, the event featured an opening ceremony with player introductions on a replica of the Golden Gate Bridge.
    Top Cosine Score: 0.7005981206893921
- `loop.py "A complete loss of rainforest cover may be caused by what type of emissions?"`
    Model Reply: Carbon emissions.
    Grounded Reply Judgement: YES; paragraph states release of carbon from vegetation, implying carbon emissions.
    Paste Context: Environmentalists are concerned about loss of biodiversity that will result from destruction of the forest, and also about the release of the carbon contained within the vegetation, which could accelerate global warming. Amazonian evergreen forests account for about 10% of the world's terrestrial primary productivity and 10% of the carbon stores in ecosystems—of the order of 1.1 × 1011 metric tonn
    Top Cosine Score: 0.6006655693054199

Judge sees gold span: no
Hand labels edited: no

What I still find shaky, in full sentences:
- So far, we're only doing simple retrieval of a single paragraph of context from a small corpus of knowledge but I'm guessing that in a production system things will get complicated quickly (for example if a user uses a chatbox to ask "Was my order from last week sent to my home address or office address?", we would need to determine the user's information, their saved addresses, fetch the orders from last week, etc. Maybe we even need a model to figure out all of the "details" we need to retrieve and then run those retrievals in parallel?)