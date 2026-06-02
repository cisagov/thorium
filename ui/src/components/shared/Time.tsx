import React, { Fragment } from 'react';

interface TimeProps {
  children: string;
  className?: string; // custom className pass through
  verbose?: boolean; // full date string
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
