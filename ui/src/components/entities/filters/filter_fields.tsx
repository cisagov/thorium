import React, { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Button, ButtonToolbar, Col, Form, Row } from 'react-bootstrap';
import 'react-datepicker/dist/react-datepicker.css';

// project imports
import { OverlayTipRight, Subtitle } from '@components';
import { decodeParamsToFilters, encodeFiltersToParams } from './params';
import { safeDateToStringConversion } from '@utilities';
import { RequestTags, FilterTypes, Filters } from '@models';
import { FilterGroups } from './groups';
import { FilterTagDisplayKeys, FilterTagsField } from './tags';
import { FilterDatePicker } from './date';
import { clearInvalidTags, DEFAULT_HIDE_TAG_KEYS, getLimitOptions } from './utilities';

interface FilterWindowProps {
  onChange: (filters: Filters) => void; // call back to change filters
  disabled: boolean; // whether changes to filters are disabled
  groups: Array<string>; // the groups a user can select from
  exclude: FilterTypes[];
  show: boolean; // whether filters are visible
}

export const FilterFields: React.FC<FilterWindowProps> = ({ disabled, onChange, groups, exclude, show }) => {
  const [filters, setFilters] = useState<Filters>({});
  const [searchParams, setSearchParams] = useSearchParams();
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
      case 'hideTags':
      case 'tags_case_insensitive':
      default:
        newFilters[key] = value;
    }
    setFilters(newFilters);
  }

  // get filters and user groups url params on initial page load
  // we do this after userInfo changes so we know a user's group membership
  const params = useMemo(() => searchParams.entries(), [searchParams]);
  useEffect(() => {
    readFilterParams();
  }, [groups, params]);

  const updateBrowsingFilters = (): void => {
    const newFilters = clearInvalidTags(filters);
    setFilters(newFilters);
    setSearchParams(encodeFiltersToParams(newFilters));
    onChange(newFilters);
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
    const newFilters: Filters = { hideTags: DEFAULT_HIDE_TAG_KEYS };
    setSearchParams(encodeFiltersToParams(newFilters));
    setFilters(newFilters);
    onChange(newFilters);
  };

  // we must force rerender when not shown to get TagsSelect to calculate the width of tag inputs correctly
  if (!show) return null;
  return (
    <>
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
                selected={filters.groups ? filters.groups.toSorted() : []}
                options={groups ? groups.toSorted() : []}
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
              <FilterTagsField
                selected={filters.tags}
                disabled={disabled}
                onChange={(newTags: RequestTags) => updateFilters('tags', newTags)}
              />
            </Col>
          </Row>
          <Row>
            <Col className="d-flex justify-content-center mt-3">
              <Subtitle>Hide Tags</Subtitle>
            </Col>
          </Row>
          <Row className="mt-2">
            <Col className="d-flex justify-content-center">
              <FilterTagDisplayKeys
                selected={filters.hideTags ? filters.hideTags.toSorted() : []}
                options={DEFAULT_HIDE_TAG_KEYS}
                onChange={(excludeKeys) => updateFilters('hideTags', excludeKeys)}
                disabled={disabled}
              />
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
                    onChange={() => updateFilters('tags_case_insensitive', !filters.tags_case_insensitive)}
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
                style={{ maxWidth: '85px' }}
                className="m-0"
                value={filters.limit ? filters.limit : 25}
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
    </>
  );
};
