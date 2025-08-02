import React, { useEffect, useState, Fragment, JSX } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Button, ButtonToolbar, Card, Col, Form, Row } from 'react-bootstrap';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { FaFilter } from 'react-icons/fa';

// project imports
import { OverlayTipRight, Subtitle, SelectGroups, SelectableDictionary, Title, OverlayTipLeft, SelectInputArray } from '@components';
import { safeDateToStringConversion, safeStringToDateConversion } from '@utilities';
import { FilterTypes, Filters, FilterTags, Entities } from '@models';
import styled from 'styled-components';

// default number of results to render when listing files
export const DEFAULT_LIST_LIMIT = 10;

// encode filters to search params
const encodeFiltersToParams = (filters: Filters) => {
  const encodedFilters = [];
  // encode limit
  if (filters.limit) {
    encodedFilters.push(`limit=${encodeURIComponent(filters.limit)}`);
  }
  // encode groups
  if (filters.groups) {
    if (Array.isArray(filters.groups)) {
      filters.groups.map((group) => {
        encodedFilters.push(`groups=${encodeURIComponent(group)}`);
      });
    } else {
      encodedFilters.push(`groups=${encodeURIComponent(filters.groups)}`);
    }
  }
  // encode nested tags
  for (const key in filters.tags) {
    filters.tags[key].map((value) => {
      encodedFilters.push(`tags[${encodeURIComponent(key)}]=${encodeURIComponent(value)}`);
    });
  }
  if (filters.hasOwnProperty('start')) {
    encodedFilters.push(`start=${filters.start}`);
  }
  if (filters.hasOwnProperty('end')) {
    encodedFilters.push(`end=${filters.end}`);
  }
  // Join all parameters with '&' to form the query string
  return encodedFilters.join('&');
};

// decode search params to filters
const decodeParamsToFilters = (searchParams: URLSearchParams) => {
  const params: Filters = {};
  const tags: FilterTags = {};
  // Iterate over each search parameter
  for (const [key, value] of searchParams.entries()) {
    // skip empty values
    if (value == '') {
      continue;
    }
    // parse tags list
    if (key.startsWith('tags[')) {
      // break up tags keys from tags prefix
      const keyTokens = key.split(/\[|\]/).filter(Boolean);
      if (keyTokens.length == 2) {
        if (tags.hasOwnProperty(keyTokens[1])) {
          // don't save duplicate tags
          if (!tags[keyTokens[1]].includes(value)) {
            tags[keyTokens[1]].push(value);
          }
        } else {
          tags[keyTokens[1]] = [value];
        }
      }
      // parse groups list for submission group membership
    } else if (key == 'groups') {
      if ('groups' in params && params.groups) {
        params.groups.push(value);
      } else {
        params['groups'] = [value];
      }
      // put all else keys as single key/value pairs in params
    } else {
      function updateFilterField<T extends keyof Filters>(field: T, value: Filters[T]): void {
        params[field] = value;
      }
      updateFilterField(key as keyof Filters, value);
    }
  }
  if (Object.keys(tags).length > 0) {
    params['tags'] = tags;
  }
  return params;
};

// get all possible limit options including the current value
function getLimitOptions(currentLimit: number): Array<number> {
  // add limit to limit options if it is not one of the defaults
  const limitOptions = [10, 25, 50, 100];
  if (currentLimit != 0 && !limitOptions.includes(currentLimit)) {
    limitOptions.push(currentLimit);
    return limitOptions.sort(function (a, b) {
      return a - b;
    });
  }
  return limitOptions;
}

interface FilterGroupsProps {
  selected: string[]; // array of selected group names
  options: string[]; // array of group name options
  onChange: (groups: string[]) => void; // set groups callback
  disabled: boolean; // disable changes to groups
}

const FilterDiv = styled.div`
  width: 70%;
`;

const FilterGroups: React.FC<FilterGroupsProps> = ({ selected, options, onChange, disabled }) => {
  const [groups, setGroups] = useState(selected.sort());
  return (
    <FilterDiv>
      <SelectInputArray
        defaultMessage="Select a group"
        disabled={disabled}
        isCreatable={false}
        options={options}
        values={groups.sort()}
        onChange={(newGroups: string[]) => {
          setGroups(newGroups);
          onChange(newGroups);
        }}
      />
    </FilterDiv>
  );
};

interface FilterDateProps {
  max?: string | Date | null | undefined;
  min?: string | Date | null | undefined;
  selected: string | Date | null | undefined;
  disabled: boolean;
  onChange: (date: Date | null) => void;
}

export const FilterDatePicker: React.FC<FilterDateProps> = ({ max = null, min = null, selected = null, disabled, onChange }) => {
  let safeMax: Date | undefined = undefined;
  let safeMin: Date | undefined = undefined;
  let safeSelected: Date | undefined = undefined;
  if (max && typeof max == 'string') {
    const maxDate = safeStringToDateConversion(max);
    if (maxDate) {
      safeMax = maxDate;
    }
  } else if (max && max instanceof Date) {
    safeMax = max;
  }
  if (min && typeof min == 'string') {
    const minDate = safeStringToDateConversion(min);
    if (minDate) {
      safeMin = minDate;
    }
  } else if (min && min instanceof Date) {
    safeMin = min;
  }
  if (selected && typeof selected == 'string') {
    const selectedDate = safeStringToDateConversion(selected);
    if (selectedDate) {
      safeSelected = selectedDate;
    }
  } else if (selected && selected instanceof Date) {
    safeSelected = selected;
  }

  return (
    <DatePicker
      //className="date-picker-input"
      isClearable={true}
      maxDate={safeMax}
      minDate={safeMin}
      selected={safeSelected}
      disabled={disabled}
      onChange={(date) => onChange(date instanceof Date ? date : null)}
    />
  );
};

const convertTagsObjectArrayToTags = (entries: TagObject[]): FilterTags => {
  // update tag object list
  const tags: FilterTags = {};
  entries.map((tag) => {
    if (tag.key == '' || tag.value == '') {
      return;
    }
    if (tag.key in tags) {
      // don't send duplicate tags
      if (!tags[tag.key].includes(tag.value)) {
        tags[tag.key].push(tag.value);
      }
    } else {
      tags[tag.key] = [tag.value];
    }
  });
  return tags;
};

interface FilterTagsProps {
  selected: FilterTags | null | undefined;
  disabled: boolean;
  onChange: (tags: FilterTags) => void;
}

interface TagObject {
  key: string;
  value: string;
}

const FilterTagsField: React.FC<FilterTagsProps> = ({ selected, onChange, disabled }) => {
  const [tags, setTags] = useState({});

  useEffect(() => {
    const tagObjectList: TagObject[] = [];
    // convert object tags into a list of individual tag objects
    if (selected) {
      Object.keys(selected).map((tagKey: string) => {
        selected[tagKey].map((tagValue: string) => {
          tagObjectList.push({ key: tagKey, value: tagValue });
        });
      });
    }
    setTags(tagObjectList);
  }, []);

  return (
    <FilterDiv>
      <SelectableDictionary
        disabled={disabled}
        entries={tags}
        setEntries={(tags: TagObject[]) => {
          onChange(convertTagsObjectArrayToTags(tags));
          setTags(tags);
        }}
        keys={null}
        deleted={null}
        setDeleted={void 0}
        trim={true}
        keyPlaceholder={'key'}
        valuePlaceholder={'value'}
      />
    </FilterDiv>
  );
};

interface BrowsingFiltersProps {
  onChange: (filters: Filters) => void; // call back to change filters
  disabled?: boolean; // whether changes to filters are disabled
  title?: string; // name of entity type being listed
  groups: Array<string>; // the groups a user can select from
  exclude?: FilterTypes[];
  creatable?: boolean; // link to create page with button
  kind?: Entities;
}

export const BrowsingFilters: React.FC<BrowsingFiltersProps> = ({
  onChange,
  groups,
  disabled = false,
  title = null,
  exclude = [],
  kind,
  creatable = false,
}) => {
  const navigate = useNavigate();
  const [filters, setFilters] = useState<Filters>({});
  const [searchParams, setSearchParams] = useSearchParams();
  // show filters or don't
  const [hideFilters, setHideFilters] = useState(true);
  // get the list of possible limit options that includes any custom values
  const limitOptions = getLimitOptions(filters.limit ? filters.limit : 0);
  // current date is the latest you can filter through
  const maxDate = new Date();

  function updateFilters<T extends keyof Filters>(key: T, value: Filters[T] | null): void {
    const newFilters = structuredClone(filters);
    // support clearing of filters
    if (value == null) {
      delete newFilters[key];
      setFilters(newFilters);
      return;
    }
    // reformat fields to work with request format
    switch (key) {
      case 'limit':
        // limit updates the search without applying any pending filter changes
        const newAppliedFilters = structuredClone(filters);
        newAppliedFilters[key] = value;
        newFilters[key] = value;
        onChange(newAppliedFilters);
        break;
      case 'end':
      case 'start':
      case 'groups':
      case 'tags':
      case 'tags_case_insensitive':
      default:
        newFilters[key] = value;
    }
    setFilters(newFilters);
  }

  // get filters and user groups url params on initial page load
  // we do this after userInfo changes so we know a user's group membership
  useEffect(() => {
    readFilterParams();
  }, [groups]);

  const updateBrowsingFilters = (): void => {
    setSearchParams(encodeFiltersToParams(filters));
    onChange(filters);
  };

  // read filter values from url search query
  const readFilterParams = (): void => {
    // get filters from query params
    const paramFilters: Filters = decodeParamsToFilters(searchParams);
    setFilters(paramFilters);
    onChange(paramFilters);
  };

  // reset all filters and get updated list from API
  const resetFilters = () => {
    const newFilters: Filters = {};
    setSearchParams(encodeFiltersToParams(newFilters));
    setFilters(newFilters);
    onChange(newFilters);
  };

  const submitFilterForm = (event: React.KeyboardEvent<HTMLElement>) => {
    // apply filters when enter is clicked, otherwise ignore
    if (event.key === 'Enter') {
      updateBrowsingFilters();
    }
  };

  return (
    <Fragment>
      <Row>
        <Col />
        <Col className="d-flex justify-content-center">
          {title && <Title>{title}</Title>}
          <OverlayTipRight tip={`${hideFilters ? 'Expand' : 'Hide'} filters`}>
            {/* @ts-ignore*/}
            <Button variant="" className="mt-3 clear-btn" onClick={() => setHideFilters(!hideFilters)}>
              <FaFilter size="18" />
            </Button>
          </OverlayTipRight>
        </Col>
        <Col className="d-flex justify-content-end">
          {creatable && (
            <OverlayTipLeft tip={`Create a new ${kind}.`}>
              <Button className="ok-btn my-3" variant="" disabled={disabled} onClick={() => navigate(`/create/${kind?.toLowerCase()}`)}>
                <b>+</b>
              </Button>
            </OverlayTipLeft>
          )}
        </Col>
      </Row>
      {!hideFilters && (
        <Card className="panel" onKeyDown={(event) => submitFilterForm(event)}>
          {!exclude.includes(FilterTypes.Groups) && (
            <>
              <Row>
                <Col className="d-flex justify-content-center mt-3">
                  <Subtitle>Groups</Subtitle>
                </Col>
              </Row>
              <Row className="mt-2">
                <Col className="d-flex justify-content-center">
                  <FilterGroups
                    selected={filters.groups ? filters.groups : []}
                    options={groups ? groups : []}
                    onChange={(groups) => updateFilters('groups', groups)}
                    disabled={disabled}
                  />
                </Col>
              </Row>
            </>
          )}
          {!exclude.includes(FilterTypes.Tags) && (
            <>
              <Row className="my-2">
                <Col className="d-flex justify-content-center">
                  <Subtitle>Tags</Subtitle>
                </Col>
              </Row>
              <Row>
                <Col className="d-flex justify-content-center">
                  <FilterTagsField disabled={disabled} selected={filters.tags} onChange={(tags) => updateFilters('tags', tags)} />
                </Col>
              </Row>
            </>
          )}
          {!exclude.includes(FilterTypes.Tags) && !exclude.includes(FilterTypes.TagsCaseInsensitive) && (
            <>
              <Row className="my-2">
                <Col className="d-flex justify-content-end align-items-center">
                  <Subtitle>Case-insensitive</Subtitle>
                </Col>
                <Col className="d-flex justify-content-start align-items-center">
                  <Form.Group>
                    <OverlayTipRight tip={`Match on tags regardless of case`}>
                      <Form.Check
                        type="switch"
                        id="case-insensitive"
                        label=""
                        checked={filters.tags_case_insensitive}
                        onChange={(e) => updateFilters('tags_case_insensitive', !filters.tags_case_insensitive)}
                      />
                    </OverlayTipRight>
                  </Form.Group>
                </Col>
              </Row>
            </>
          )}
          {!exclude.includes(FilterTypes.End) && (
            <>
              <Row className="mt-3">
                <Col xs={4} md={6} className="d-flex justify-content-end">
                  <Subtitle className="mt-2">Oldest</Subtitle>
                </Col>
                <Col className="d-flex justify-content-start">
                  <FilterDatePicker
                    max={filters.start ? filters.start : maxDate}
                    selected={filters.end}
                    disabled={disabled}
                    onChange={(date) => updateFilters('end', safeDateToStringConversion(date))}
                  />
                </Col>
              </Row>
            </>
          )}
          {!exclude.includes(FilterTypes.Start) && (
            <>
              <Row className="mt-1">
                <Col xs={4} md={6} className="d-flex justify-content-end">
                  <Subtitle className="mt-2">Newest</Subtitle>
                </Col>
                <Col className="d-flex justify-content-start">
                  <FilterDatePicker
                    max={maxDate}
                    min={filters.end}
                    selected={filters.start}
                    disabled={disabled}
                    onChange={(date) => updateFilters('start', safeDateToStringConversion(date))}
                  />
                </Col>
              </Row>
            </>
          )}
          <Row className="m-3">
            <Col className="d-flex justify-content-center">
              <ButtonToolbar>
                {/* @ts-ignore */}
                <Button
                  className="ok-btn"
                  disabled={disabled}
                  onClick={() => {
                    updateBrowsingFilters();
                  }}
                >
                  Apply
                </Button>
                <Button
                  className="primary-btn"
                  disabled={disabled}
                  onClick={() => {
                    resetFilters();
                  }}
                >
                  Clear
                </Button>
              </ButtonToolbar>
            </Col>
          </Row>
          {!exclude.includes(FilterTypes.Limit) && (
            <>
              <Row className="mb-3 mt-1">
                <Col className="d-flex justify-content-center">
                  <Form.Select
                    className="m-0 limit-select"
                    value={filters.limit}
                    disabled={disabled}
                    onChange={(e) => {
                      updateFilters('limit', parseInt(e.target.value));
                    }}
                  >
                    {limitOptions.map((count, idx) => (
                      <option key={`limit_${count}_${idx}`} value={count}>
                        {count}
                      </option>
                    ))}
                  </Form.Select>
                </Col>
              </Row>
            </>
          )}
        </Card>
      )}
    </Fragment>
  );
};
