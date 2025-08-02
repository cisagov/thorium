import React, { Fragment } from 'react';

interface TimeProps {
  children: string;
  className?: string; // custom className pass through
  verbose?: boolean; // full date string
}

export const Time: React.FC<TimeProps> = ({ children, verbose }) => {
  let date;
  let fullTime;
  let time;
  try {
    [date, fullTime] = children.split('T');
    time = fullTime.split('.')[0];
  } catch (e) {
    return <Fragment>{children}</Fragment>;
  }
  if (verbose) {
    return (
      <Fragment>
        <i>{'on '}</i>
        {date}
        <i>{' at '}</i>
        {time}
      </Fragment>
    );
  }

  return <Fragment>{date + ' ' + time}</Fragment>;
};
