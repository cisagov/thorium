import React, { Fragment } from 'react';

// spec: ./Time.spec.md

interface TimeProps {
  children: string;
  className?: string;
  /** Renders the long-form "on <date> at <time>" string instead of the compact "<date> <time>". */
  verbose?: boolean;
}

const Time: React.FC<TimeProps> = ({ children, verbose }) => {
  let date: string;
  let fullTime: string | undefined;
  let time: string;
  try {
    [date, fullTime] = children.split('T');
    if (!fullTime) {
      return <Fragment>{children}</Fragment>;
    }
    time = fullTime.split('.')[0];
  } catch {
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

export default Time;
