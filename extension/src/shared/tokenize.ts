/**
 * Truncation, which the two tokenizer implementations do differently.
 *
 * Python's `tokenizers` treats `max_length` as a budget for the *content*: it truncates the
 * content, then applies the post-processor, so the sequence always ends with `[SEP]`.
 * transformers.js truncates the already-post-processed sequence by slicing, which keeps
 * `[CLS]` at the front and cuts `[SEP]` off the end.
 *
 * The two agree on every token except the last:
 *
 *   python  [CLS] c0 c1 ... c509 [SEP]
 *   js      [CLS] c0 c1 ... c509 c510
 *
 * That single token is worth ~0.42 of logit on a real chunk, because the model pools from
 * `[CLS]` over a sequence whose terminator has gone missing.
 *
 * This is not an edge case. `chunk_block` caps chunks at 400 words and scope.md 4.3 notes
 * that at ~1.25-1.35 BPE tokens per word a 400-word window is ~500-540 tokens -- so the
 * longest chunks on any real page truncate, routinely.
 *
 * Pinned by the truncated cases in fixtures/tokenize.json.
 */

/**
 * Re-truncate a post-processed sequence the way Python's `tokenizers` would.
 *
 * Takes the *untruncated* ids and mask so nothing depends on transformers.js's own
 * truncation, and returns at most `maxLength` tokens ending in `sepId`.
 */
export function truncateLikeTokenizers(
  ids: readonly number[],
  mask: readonly number[],
  maxLength: number,
  sepId: number,
): { ids: number[]; mask: number[] } {
  if (ids.length <= maxLength) {
    return { ids: [...ids], mask: [...mask] };
  }
  const cut = ids.slice(0, maxLength);
  const cutMask = mask.slice(0, maxLength);
  // The post-processor's closing token, which slicing removed.
  cut[maxLength - 1] = sepId;
  cutMask[maxLength - 1] = 1;
  return { ids: cut, mask: cutMask };
}
