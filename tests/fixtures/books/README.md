# Reference books for translation testing

Full-length public domain novels used to evaluate translation quality and
formatting preservation on realistic EPUB markup (long documents, dense inline
tags, poetry, figures). They complement `../translation_sampler.epub`, which is
small and synthetic.

| File | Work | Words | Notable markup |
|------|------|-------|----------------|
| `alice_in_wonderland.epub` | Lewis Carroll, *Alice's Adventures in Wonderland* (1865), illustrated by John Tenniel | ~28,000 | Heavy `<em>` emphasis, verse in `<blockquote>` with `<br>`, 42 `<figure>` blocks, wordplay |
| `picture_of_dorian_gray.epub` | Oscar Wilde, *The Picture of Dorian Gray* (1891) | ~80,000 | Long literary prose, `<abbr>`, `<i>` for foreign terms and titles, 20 chapters |

## Source and license

Both files come from [Standard Ebooks](https://standardebooks.org/):

- https://standardebooks.org/ebooks/lewis-carroll/alices-adventures-in-wonderland/john-tenniel
- https://standardebooks.org/ebooks/oscar-wilde/the-picture-of-dorian-gray

The texts and artwork are in the public domain. Standard Ebooks contributors
release their markup and editorial work under the
[CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)
(see `epub/text/uncopyright.xhtml` inside each file).

## Modifications

To keep the repository small, every PNG and JPEG inside the EPUBs was replaced
with a 1x1 placeholder image of the same format and file name. All XHTML, CSS,
OPF and navigation files are byte-identical to the originals, so the markup
(including `<img>` and `<figure>` elements) is unchanged.
