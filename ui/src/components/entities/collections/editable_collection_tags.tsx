import { useEffect, useMemo, useState, useRef } from 'react';
import CreatableSelect from 'react-select/creatable';
import type { ActionMeta, MultiValue } from 'react-select';
import { Row, Col } from 'react-bootstrap';

import { AlertBanner } from '@components';
import { createReactSelectStyles } from '@utilities';
import { CollectionTags } from 'models';

type TagOption = {
  value: string; // displayed "k: v"
  label: string;
  thoriumTag: { key: string; value: string | null };
};

const tagStyles = createReactSelectStyles('White', 'rgb(160, 162, 163)');

const parseRawTag = (raw: string): { key: string; value: string | null } => {
  const trimmed = (raw ?? '').trim();
  if (!trimmed) return { key: '', value: null };

  // Support ":" or "=" like the file editor
  const delim = trimmed.includes(':') ? ':' : trimmed.includes('=') ? '=' : null;
  if (!delim) return { key: trimmed, value: null };

  const key = trimmed.split(delim, 1)[0].trim();
  const value = trimmed.substring(key.length + 1).trim();
  return { key, value: value.length ? value : null };
};

const toOptions = (tags: CollectionTags): TagOption[] => {
  const opts: TagOption[] = [];
  Object.keys(tags ?? {})
    .sort()
    .forEach((k) => {
      (tags[k] ?? [])
        .slice()
        .sort()
        .forEach((v) => {
          const s = `${k}: ${v}`;
          opts.push({ value: s, label: s, thoriumTag: { key: k, value: v } });
        });
    });
  return opts;
};

const toTagMap = (selected: TagOption[]): CollectionTags => {
  const map: CollectionTags = {};
  selected.forEach((opt) => {
    const k = opt?.thoriumTag?.key?.trim();
    const v = opt?.thoriumTag?.value?.trim?.() ?? opt?.thoriumTag?.value;
    if (!k || !v) return; // ignore invalids
    if (!map[k]) map[k] = [];
    if (!map[k].includes(v)) map[k].push(v);
  });

  // keep stable ordering
  Object.keys(map).forEach((k) => map[k].sort());
  return map;
};

type Props = {
  value: CollectionTags; // pendingCollection.metadata.Collection.collection_tags
  onChange: (next: CollectionTags) => void; // calls updatePendingMeta('collection_tags', next)
};

const EditableCollectionTags = ({ value, onChange }: Props) => {
  const initialSelected = useMemo(() => toOptions(value ?? {}), [value]);

  const [selected, setSelected] = useState<TagOption[]>(initialSelected);
  const [invalidKeys, setInvalidKeys] = useState<string[]>([]);

  // Tracks the last (valid) tag map we emitted to the parent. This prevents the
  // "prop echo" from resetting local invalid state immediately after typing.
  const lastEmittedRef = useRef<string>('');

  const emit = (nextSelected: TagOption[]) => {
    const nextMap = toTagMap(nextSelected);
    lastEmittedRef.current = JSON.stringify(nextMap);
    onChange(nextMap);
  };

  // keep local state synced if parent changes (e.g., cancel/reset editing)
  useEffect(() => {
    const incoming = JSON.stringify(value ?? {});
    if (incoming === lastEmittedRef.current) return;

    setSelected(initialSelected);
    setInvalidKeys([]);
  }, [value, initialSelected]);

  const handleChange = (newSelected: MultiValue<TagOption>, meta: ActionMeta<TagOption>) => {
    const nextSelected = (newSelected as TagOption[]) ?? [];
    let nextInvalid = [...invalidKeys];

    // When user "creates" a new option, we need to parse and mark invalids
    if (meta.action === 'create-option' && meta.option) {
      const parsed = parseRawTag(meta.option.value);

      // attach thoriumTag so future remove/select operations have key/value
      (meta.option as any).thoriumTag = parsed;

      // invalid if missing key or value
      if (!parsed.key || !parsed.value) {
        if (parsed.key && !nextInvalid.includes(parsed.key)) nextInvalid.push(parsed.key);
      }
    }

    // If user removes/clears, also remove any matching invalid key markers
    if (meta.action === 'remove-value' && meta.removedValue?.thoriumTag) {
      const k = meta.removedValue.thoriumTag.key;
      const v = meta.removedValue.thoriumTag.value;
      if (k && !v) nextInvalid = nextInvalid.filter((x) => x !== k);
    }
    if (meta.action === 'clear' && meta.removedValues?.length) {
      meta.removedValues.forEach((rv) => {
        const k = rv?.thoriumTag?.key;
        const v = rv?.thoriumTag?.value;
        if (k && !v) nextInvalid = nextInvalid.filter((x) => x !== k);
      });
    }

    setInvalidKeys(nextInvalid);
    setSelected(nextSelected);

    // Only propagate valid tags upward; invalids are ignored
    emit(nextSelected);
  };

  return (
    <>
      <CreatableSelect
        isMulti
        isSearchable
        isClearable
        value={selected}
        styles={tagStyles}
        onChange={handleChange}
        // We don’t need "options" here because collection tags are fully user-defined;
        // but you can provide suggestions if you want later.
        options={selected}
        noOptionsMessage={({ inputValue }) =>
          inputValue ? inputValue : `Create a tag by typing "key:value" or "key=value" and pressing enter.`
        }
      />
      {invalidKeys.length > 0 && (
        <Row className="mt-2">
          <Col>
            <AlertBanner
              prefix="Collection tags"
              errorStatus={`Invalid tags: ${invalidKeys.join(', ')} (must include both key and value; these will be ignored)`}
              variant="danger"
            />
          </Col>
        </Row>
      )}
    </>
  );
};

export default EditableCollectionTags;
