; Maolan Generate Installer
; Run with: makensis.exe installer.nsi
; Requires all binaries and DLLs to be staged in C:\maolan-staging\generate

Unicode true

!include "MUI2.nsh"
!include "LogicLib.nsh"

!ifndef MAOLAN_GENERATE_VERSION
!define MAOLAN_GENERATE_VERSION "0.0.11"
!endif

!ifndef MAOLAN_GENERATE_PRODUCT_VERSION
!define MAOLAN_GENERATE_PRODUCT_VERSION "${MAOLAN_GENERATE_VERSION}.0"
!endif

;--------------------------------
; General
;--------------------------------
Name "Maolan Generate"
OutFile "maolan-generate-setup.exe"
InstallDir "$LOCALAPPDATA\Maolan\bin"
InstallDirRegKey HKCU "Software\MaolanGenerate" "InstallDir"
RequestExecutionLevel user

;--------------------------------
; Version Info
;--------------------------------
VIProductVersion "${MAOLAN_GENERATE_PRODUCT_VERSION}"
VIAddVersionKey "ProductName" "Maolan Generate"
VIAddVersionKey "ProductVersion" "${MAOLAN_GENERATE_VERSION}"
VIAddVersionKey "FileVersion" "${MAOLAN_GENERATE_VERSION}"
VIAddVersionKey "FileDescription" "Maolan AI Music Generation"
VIAddVersionKey "LegalCopyright" "BSD-2-Clause"

;--------------------------------
; Interface Settings
;--------------------------------
!define MUI_ABORTWARNING
!ifndef MAOLAN_GENERATE_ICON
!define MAOLAN_GENERATE_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!endif
!define MUI_ICON "${MAOLAN_GENERATE_ICON}"
!define MUI_UNICON "${MAOLAN_GENERATE_ICON}"

;--------------------------------
; Pages
;--------------------------------
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

;--------------------------------
; Languages
;--------------------------------
!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Installer Sections
;--------------------------------
Section "Install"
    SetOutPath "$INSTDIR"

    ; Copy all staged binaries and DLLs
    File "C:\maolan-staging\generate\*.*"

    ; Run VC++ Redistributable installer
    ExecWait '"$INSTDIR\vc_redist.x64.exe" /install /quiet /norestart' $0
    Delete "$INSTDIR\vc_redist.x64.exe"

    ; Store installation folder
    WriteRegStr HKCU "Software\MaolanGenerate" "InstallDir" $INSTDIR

    ; Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Add to Add/Remove Programs
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "DisplayName" "Maolan Generate"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "DisplayVersion" "${MAOLAN_GENERATE_VERSION}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "Publisher" "Maolan Team"
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "NoModify" 1
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "NoRepair" 1

    ; Create Start Menu shortcuts
    CreateDirectory "$SMPROGRAMS\Maolan Generate"
    CreateShortcut "$SMPROGRAMS\Maolan Generate\Maolan Generate.lnk" "$INSTDIR\maolan-generate.exe" "" "$INSTDIR\maolan-generate.exe" 0
    CreateShortcut "$SMPROGRAMS\Maolan Generate\Uninstall.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0

    ; Create desktop shortcut
    CreateShortcut "$DESKTOP\Maolan Generate.lnk" "$INSTDIR\maolan-generate.exe" "" "$INSTDIR\maolan-generate.exe" 0
SectionEnd

;--------------------------------
; Uninstaller Section
;--------------------------------
Section "Uninstall"
    Delete "$INSTDIR\maolan-generate.exe"
    Delete "$INSTDIR\Uninstall.exe"

    Delete "$SMPROGRAMS\Maolan Generate\Maolan Generate.lnk"
    Delete "$SMPROGRAMS\Maolan Generate\Uninstall.lnk"
    RMDir "$SMPROGRAMS\Maolan Generate"

    Delete "$DESKTOP\Maolan Generate.lnk"

    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate"
    DeleteRegKey HKCU "Software\MaolanGenerate"

    RMDir "$INSTDIR"
SectionEnd
