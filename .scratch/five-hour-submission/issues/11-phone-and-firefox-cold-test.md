# 11: Phone and Firefox cold test

**What to build:** Evidence that the live URL works for someone who is not the owner: a cold load on a phone and in Firefox, with any failures fixed or recorded. This is the only browser evidence in the five-hour build (no Node smoke test).

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 05 (Build workflow and first live deploy)

**Status:** ready-for-human

- [ ] Open the live URL on a phone and in Firefox from a cold cache, on a connection that is not the one used for development.
- [ ] Record time to the first usable screen, whether one complete Reading and one with a missing input assess correctly, and whether the in-app golden check (if built) passed.
- [ ] A first load over 60 s, or any failure, is recorded in the ticket's comments with the device, browser and connection; failures that can be fixed within the stop rule get a new ticket.
- [ ] The results are available to the README ticket as the stated, limited browser evidence (one device and browser).
