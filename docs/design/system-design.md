# System design

What is built for this submission (solid boxes and arrows) and what a production system would add
(dashed boxes and arrows, described only, nothing of it exists in the repository). Linked from the
[README](../../README.md). The scope that decides what is built is
[ADR 0012](../adr/0012-five-hour-submission-scope.md).

```mermaid
flowchart LR
    classDef built fill:#e8f1ff,stroke:#2b5fb4,stroke-width:1.5px,color:#111
    classDef prod fill:#fff,stroke:#888,stroke-width:1.5px,stroke-dasharray:6 4,color:#444

    subgraph ING["1. Ingestion"]
        CSV["data/raw/ai4i2020.csv<br/>(committed, 10,000 Readings)"]:::built
        LOAD["data/ingest + clean<br/>schema and integrity checks"]:::built
        FEED["Plant sensor feed or historian<br/>(live Readings)"]:::prod
        VALID["Input contract checks and<br/>ingestion alerts"]:::prod
        CSV --> LOAD
        FEED -.-> VALID
    end

    subgraph TRN["2. Training (just train, and in CI)"]
        SPLIT["Stratified 70/30 split<br/>seed from configs/"]:::built
        TRAIN["Train partition only:<br/>repeated 5x5 CV, Operating threshold,<br/>fit random forest Pipeline"]:::built
        UNIT["Five-file artifact unit<br/>Pipeline, OperatingPoint, missing-data policy,<br/>operating-metrics, manifest (SHA-256)"]:::built
        HOLD["Holdout scored once<br/>(just holdout)"]:::built
        REPORT["reports/holdout-metrics.json<br/>(committed, with intervals)"]:::built
        RETRAIN["Scheduled or triggered retraining<br/>on new labelled Readings"]:::prod
        FRESH["Candidate scored on a fresh holdout"]:::prod
        SHADOW["Shadow period beside the live model"]:::prod
        REG["Model registry and<br/>human approval to promote"]:::prod
        LOAD --> SPLIT --> TRAIN --> UNIT
        SPLIT -->|"30 % Holdout, never fitted on"| HOLD --> REPORT
        UNIT --> HOLD
        RETRAIN -.-> FRESH -.-> SHADOW -.-> REG
    end

    subgraph SRV["3. Serving"]
        BUILD["pages.yml (on TRIGGER_CI):<br/>train, check report hash, build wheel,<br/>export marimo WebAssembly site,<br/>contract checks"]:::built
        PAGES["GitHub Pages<br/>static site, unit published beside it"]:::built
        INF["inference/ in the browser (Pyodide):<br/>load and verify unit, assess one Reading,<br/>Presentation (the words)"]:::built
        APP["marimo_app.py<br/>one screen, imports no logic"]:::built
        ENG(["Maintenance engineer"]):::built
        API["Hosted inference service with<br/>authentication and access control"]:::prod
        WO["Work-order or CMMS hand-off"]:::prod
        UNIT --> BUILD
        REPORT --> BUILD
        BUILD --> PAGES --> INF --> APP --> ENG
        REG -.-> API -.-> WO -.-> ENG
    end

    subgraph MON["4. Monitoring"]
        SESS["Session log: one record per Reading,<br/>no raw values (designed, not built)"]:::prod
        SERVER["Server-side assessment log"]:::prod
        MET["Refusal rate, Flag rate against expected,<br/>input drift (Jensen-Shannon, or KS on raw values)"]:::prod
        OUT["Post-outcome precision and recall<br/>(maintenance outcomes joined after a label delay)"]:::prod
        ALERT["Dashboards, alerts and<br/>retraining triggers"]:::prod
        INF -.-> SESS -.-> SERVER
        API -.-> SERVER
        SERVER -.-> MET -.-> ALERT
        WO -.-> OUT -.-> ALERT
        ALERT -.-> RETRAIN
        VALID -.-> MET
    end
```

## Reading the diagram

- **Solid, blue** boxes and arrows exist in this repository and run today. **Dashed, white** boxes and
  dashed arrows are production additions that are described only. The engineer is real; the dashed arrow
  into them is the work-order hand-off, which is not built.
- **Ingestion** is a committed CSV. The built path validates the schema and integrity of the data. A live
  feed, input contract checks and ingestion alerts are not built.
- **Training** is one command that touches only the train partition. The Holdout is scored once, from
  the CI-built artifact, and the report is committed. A scheduled retraining loop with a fresh holdout and
  a shadow period is the production counterpart (see the README's retraining strategy).
- **Serving** is a static site: Python runs in the visitor's browser (Pyodide), so there is no server. A
  hosted inference service, access control and a work-order hand-off are not built.
- **Monitoring** is not built. The refusal rate, the Flag rate against expected, input drift and the
  post-outcome precision and recall are defined in the README. A per-session log, a health tab and the Sample log are
  designed in ADRs 0008 to 0010 and deferred by ADR 0012.
