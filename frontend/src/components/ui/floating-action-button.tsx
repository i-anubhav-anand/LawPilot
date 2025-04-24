import * as React from "react"
import { cn } from "@/lib/utils"
import { Button } from "./button"
import { cva } from "class-variance-authority"

const fabVariants = cva(
  "rounded-full shadow-lg flex items-center justify-center transition-all duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2",
  {
    variants: {
      size: {
        default: "h-14 w-14",
        sm: "h-12 w-12",
        lg: "h-16 w-16",
      },
      variant: {
        default: "bg-purple-600 text-white hover:bg-purple-700",
        secondary: "bg-gray-100 text-gray-800 hover:bg-gray-200",
        error: "bg-red-600 text-white hover:bg-red-700",
      },
    },
    defaultVariants: {
      size: "default",
      variant: "default",
    },
  }
)

interface FloatingActionButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  size?: "default" | "sm" | "lg"
  variant?: "default" | "secondary" | "error"
  icon?: React.ReactNode
  className?: string
}

export function FloatingActionButton({
  size,
  variant,
  icon,
  className,
  ...props
}: FloatingActionButtonProps) {
  return (
    <Button
      className={cn(fabVariants({ size, variant }), className)}
      {...props}
    >
      {icon}
    </Button>
  )
} 