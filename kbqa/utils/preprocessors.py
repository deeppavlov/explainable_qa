import re


def get_ent_rels(query, db):
    entities = []
    rels = []
    if db == "dbpedia":
        triplets = query[query.find('{')+1:query.find('}')].strip('  . ').strip(' ').replace('  ', ' ').split(". ")
        for triplet in triplets:
            triplet_split = triplet.split(' ')
            if triplet_split[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                rels.append(triplet_split[1].strip('<').strip('>').replace("'", ''))
                if triplet_split[0] != "?uri" and triplet_split[0] != "?x":
                    entities.append(triplet_split[0].strip('<').strip('>'))
                if triplet_split[2] != "?uri" and triplet_split[2] != "?x":
                    entities.append(triplet_split[2].strip('<').strip('>'))
    elif db == "wikidata":
        query = query.replace("\n", " ").replace("   ", " ").replace("  ", " ")
        query_split = re.findall("{[ ]?(.*?)[ ]?}", query)
        if query_split:
            query_triplets = query_split[0].split(' . ')
            query_triplets = [triplet.split(' ')[:3] for triplet in query_triplets]
            query_triplets = [triplet for triplet in query_triplets
                              if len(triplet) == 3 and all([not triplet[1].startswith(rel)
                              for rel in {"wdt:P31", "wdt:P279", "wdt:P21"}])]
            for triplet in query_triplets:
                if triplet[0].startswith("wd:"):
                    entities.append(triplet[0][3:])
                if triplet[-1].startswith("wd:"):
                    entities.append(triplet[-1][3:])
                if not triplet[1].startswith("?"):
                    fnd = re.findall(r"P[\d]+", triplet[1])
                    if fnd:
                        rels.append(fnd[0])
        entities = list(set(entities))
        rels = list(set(rels))
    return entities, rels
