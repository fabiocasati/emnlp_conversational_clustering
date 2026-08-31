# Example conversations

Three complete conversational-clustering sessions from the batch reported in
the paper, each in its own directory (named by the POV the simulated oracle
pursued):

- `conversation_trace.json` — the full oracle/agent message log
  (`human` = simulated oracle, `assistant` = clustering agent), with
  per-turn tool-call statistics.
- `selected_topic.json` — the discovered POV (label + guidance) the
  conversation pursued.
- `clustering.json` — the final saved partition of the 1,000 tickets.

All items are synthetic benchmark tickets; the "human" side is the simulated
oracle described in the paper's appendix on simulating human oracles.
