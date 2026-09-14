# Corpus provenance

Large encoding-test corpus, tracked as zstd frames so the checksum
replaces the downloader.

- recorded once, the sha256 of the original bytes is the reference
- no network fetch happens at test time

## Container contract

- each file is a single zstd frame, compression level 19
- the frame header records the content size and a checksum
- readers stream through the stdlib `compression.zstd` module
- original bytes are recoverable exactly, the frames round-trip
  byte for byte against the recorded sha256

## Files

| file | upstream | sha256 (original) | original | frame |
|---|---|---|---|---|
| sqlite3.c.zst | sqlite.org amalgamation, SQLite 3.51.2 | f7d8b32e48849494f13851e13e659d70ef372c5d207b84bc71c05d1e62716305 | 9388884 B | 1736659 B |
| pg100-shakespeare.txt.zst | www.gutenberg.org/ebooks/100 | 4291cb282e90f6580fa683148f7c94a55276acb1757d725a2e84caf8c00cb9a5 | 5638525 B | 1688320 B |
| pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst | www.gutenberg.org/ebooks/4791 | 51cac1de6a32a0659ce5b1566b7ab7df04eddf4cf74effb4c73d89b43d089910 | 461155 B | 147563 B |
| pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst | www.gutenberg.org/ebooks/23950 | 9c12928d295be9f0fe82edbad829291e9a946905b15ddac2608b1194fbfe0672 | 1863697 B | 648372 B |

## Refresh ritual

- fetch the upstream artifact, verify it against the recorded sha256
  when the content is expected to be identical
- recompress at level 19 with content size and checksum in the header
- update the recorded sha256 and sizes in this file
- the encoding tests read prefixes, so any content change shifts
  expected token streams and must come with re-recorded expectations
