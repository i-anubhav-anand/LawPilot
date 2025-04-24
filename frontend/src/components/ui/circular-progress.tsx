"use client"

import * as React from "react"

interface CircularProgressProps {
  value?: number; // 0-100, make optional to allow indeterminate mode without value
  size?: number;  // size in pixels
  strokeWidth?: number;
  color?: string;
  backgroundColor?: string;
  indeterminate?: boolean; // Add support for indeterminate state
}

export function CircularProgress({
  value = 0,
  size = 20,
  strokeWidth = 2,
  color = "rgb(168, 85, 247)", // purple-500
  backgroundColor = "rgb(243, 232, 255)", // purple-100
  indeterminate = false,
}: CircularProgressProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(Math.max(value, 0), 100); // Clamp between 0-100
  const offset = circumference - (progress / 100) * circumference;
  
  // For indeterminate animation
  const [rotation, setRotation] = React.useState(0);
  
  React.useEffect(() => {
    if (indeterminate) {
      const interval = setInterval(() => {
        setRotation(prev => (prev + 5) % 360);
      }, 50);
      return () => clearInterval(interval);
    }
  }, [indeterminate]);

  return (
    <svg 
      width={size} 
      height={size} 
      viewBox={`0 0 ${size} ${size}`} 
      className={indeterminate ? "animate-spin" : "transform -rotate-90"}
      style={indeterminate ? { animationDuration: '1.5s' } : undefined}
    >
      {/* Background circle */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        strokeWidth={strokeWidth}
        stroke={backgroundColor}
        fill="none"
      />
      
      {/* Progress circle */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        strokeWidth={strokeWidth}
        stroke={color}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={indeterminate ? circumference * 0.75 : offset}
        fill="none"
        style={indeterminate 
          ? { transition: "none" } 
          : { transition: "stroke-dashoffset 0.3s ease" }
        }
      />
    </svg>
  );
} 