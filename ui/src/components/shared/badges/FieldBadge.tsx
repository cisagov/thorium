import React, { JSX } from 'react';
import { Badge } from 'react-bootstrap';
import styled from 'styled-components';

const TagBadge = styled(Badge)`
  text-wrap: pretty;
  white-space: break-spaces;
  /* break long unbroken tokens (image paths, key:value tags) so they wrap instead of overflowing */
  overflow-wrap: anywhere;
  max-width: 100%;
`;

interface FieldBadgeProps {
  field: object | Array<string> | string | number | bigint | boolean | null | undefined; // content of the badge
  color: string; // color of the badge
  className?: string; // custom class name
  noNull?: boolean; // don't return a badge when null
}

// returns a badge for a given image field based on the provided color
const FieldBadge: React.FC<FieldBadgeProps> = ({ field, color, className = '', noNull = false }) => {
  let value = field;
  // objects must be handled differently than other string values
  // string conversion of objects are [object Object] for empty and
  // nonempty strings
  if (typeof field == 'object' && field != null) {
    if ((!Array.isArray(field) && Object.keys(field).length > 0) || (Array.isArray(field) && field.length > 0 && String(field) != '')) {
      // if object (array or dict) we do no conversions
      value = field;
    } else {
      // empty array or dict will become none value in switch
      value = '';
    }
  }

  // Use true/false coloring if the value is a boolean
  switch (value) {
    case false:
      return (
        <Badge className={`bg-amber ${className}`} bg="">
          {String(value)}
        </Badge>
      );
    case true:
      return (
        <Badge className={`bg-green ${className}`} bg="">
          {String(value)}
        </Badge>
      );
    // None values from api
    case 'None':
    case undefined:
    case null:
    case '':
      if (noNull) {
        return <></>;
      }
      return (
        <Badge className={`bg-grey ${className}`} bg="">
          {String('none')}
        </Badge>
      );
    default:
      if (Array.isArray(field) && field.length) {
        // returns one badge per item in array
        const badgeArray = field.map((item, idx) => (
          <TagBadge key={idx} bg="" className={`me-1 ${className}`} style={{ backgroundColor: color }}>
            {typeof item == 'object' && field != null ? JSON.stringify(item) : String(item)}
          </TagBadge>
        ));
        return badgeArray;
      } else if (typeof field == 'object') {
        const badgeArray: JSX.Element[] = [];
        // if the object contains key/value pairs
        if (field) {
          Object.keys(field).forEach((itemKey: string, index: number) => {
            badgeArray.push(
              <TagBadge key={index} className={`me-1 ${className}`} style={{ backgroundColor: color }}>
                {`${itemKey}: ${String(field[itemKey as keyof typeof field])}`}
              </TagBadge>,
            );
          });
        }
        return badgeArray;
      } else {
        return (
          <TagBadge bg="" className={`me-1 ${className}`} style={{ backgroundColor: color }}>
            {String(field)}
          </TagBadge>
        );
      }
  }
};

export default FieldBadge;
