import requests
import json

url = "https://api.catalysis-hub.org/graphql"

query = """
query{reactions (pubId: "AlonsoStrain2023",
        first: 200, after: "",
        order: "chemicalComposition") {
        totalCount
        pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
        }
        edges {
          node {
            Equation
            sites
            id
            pubId
            dftCode
            dftFunctional
            reactants
            products
            facet
            reactionEnergy
            activationEnergy
            surfaceComposition
            chemicalComposition
            reactionSystems {
              name
              aseId
            }
          }
        }
        }

"""

response = requests.post(url, json={"query": query})
data = response.json()

with open("AlonsoStrain2023_full_dataset.json", "w") as f:
    json.dump(data["data"], f, indent=2)
